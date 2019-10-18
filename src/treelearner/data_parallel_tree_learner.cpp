/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cstring>
#include <tuple>
#include <vector>
#include <math.h>

#include "parallel_tree_learner.h"

namespace LightGBM
{

template <typename TREELEARNER_T>
DataParallelTreeLearner<TREELEARNER_T>::DataParallelTreeLearner(const Config *config)
    : TREELEARNER_T(config)
{
}

template <typename TREELEARNER_T>
DataParallelTreeLearner<TREELEARNER_T>::~DataParallelTreeLearner()
{
  pHubAllReduceT3->FastTerminate();
  pHubAllReduceSplitInfo->FastTerminate();
  pHubReduceScatter->FastTerminate();
}

/*  inline static void SumReducer(const char *src, char *dst, int type_size, comm_size_t len) {
    comm_size_t used_size = 0;
    const HistogramBinEntry* p1;
    HistogramBinEntry* p2;
    while (used_size < len) {
      // convert
      p1 = reinterpret_cast<const HistogramBinEntry*>(src);
      p2 = reinterpret_cast<HistogramBinEntry*>(dst);
      // add
      p2->cnt += p1->cnt;
      p2->sum_gradients += p1->sum_gradients;
      p2->sum_hessians += p1->sum_hessians;
      src += type_size;
      dst += type_size;
      used_size += type_size;
    }
  }


  const char *src, char *dst, int type_size, comm_size_t len) {
    comm_size_t used_size = 0;
    const std::tuple<data_size_t, double, double> *p1;
    std::tuple<data_size_t, double, double> *p2;
    while (used_size < len)
    {
      p1 = reinterpret_cast<const std::tuple<data_size_t, double, double> *>(src);
      p2 = reinterpret_cast<std::tuple<data_size_t, double, double> *>(dst);
      std::get<0>(*p2) = std::get<0>(*p2) + std::get<0>(*p1);
      std::get<1>(*p2) = std::get<1>(*p2) + std::get<1>(*p1);
      std::get<2>(*p2) = std::get<2>(*p2) + std::get<2>(*p1);
      src += type_size;
      dst += type_size;
      used_size += type_size;
    }
  }*/

void PHubTuple3Reducer(char *src, char *dst)
{
  std::tuple<data_size_t, double, double> *src_t = (std::tuple<data_size_t, double, double> *)(src);
  std::tuple<data_size_t, double, double> *dst_t = (std::tuple<data_size_t, double, double> *)(dst);
  std::get<0>(*dst_t) += std::get<0>(*src_t);
  std::get<1>(*dst_t) += std::get<1>(*src_t);
  std::get<2>(*dst_t) += std::get<2>(*src_t);
}

void PHubHistogramBinEntrySumReducer(char *src, char *dst)
{
  HistogramBinEntry *source = (HistogramBinEntry *)src;
  HistogramBinEntry *dest = (HistogramBinEntry *)dst;
  //it will be called repeatedly as more data is streamed to PHub
  dest->cnt += source->cnt;
  dest->sum_gradients += source->sum_gradients;
  dest->sum_hessians += source->sum_hessians;
}

std::string getKeyOwnershipString(int numMachines, int keysPerMachine)
{
  std::string ret = "[";
  for (int m = 0; m < numMachines; m++)
  {
    for (int i = 0; i < keysPerMachine; i++)
    {
      ret.append(std::to_string(m));
      if (m != numMachines - 1 || i != keysPerMachine - 1)
      {
        ret.append(",");
      }
    }
  }
  ret.append("]");
  return ret;
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::InitializePHub()
{
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();

  // allocate buffer for communication
  auto chunkSize = atoi(pHubGetMandatoryEnvironmemtVariable("PHubChunkElementSize").c_str());
  pHubChunkSize = chunkSize;
  size_t numbin = RoundUp(this->train_data_->NumTotalBin(), chunkSize);
  size_t buffer_size = numbin * sizeof(HistogramBinEntry);
  reduceScatterPerNodeBufferSize = buffer_size;
  auto total_buffer_size = buffer_size * num_machines_;

  pHubBackingBufferForReduceScatter.resize(total_buffer_size);
  reduceScatterNodeStartingAddress.resize(num_machines_);
  reduceScatterNodeStartingKey.resize(num_machines_);

  for (int i = 0; i < num_machines_; i++)
  {
    reduceScatterNodeByteCounters.push_back(std::make_unique<std::atomic<int>>(0));
    reduceScatterNodeStartingAddress.at(i) = pHubBackingBufferForReduceScatter.data() + i * buffer_size;
    reduceScatterNodeStartingKey.at(i) = i * (numbin / chunkSize);
  }
  //void getChunkedInformationGivenBuffer(
  //void *ptr,
  //size_t elements,
  //size_t elementSize,
  //int chunkElementCount,
  //std::vector<size_t> &counts,
  //std::vector<size_t> &bytes,
  //std::vector<void *> keyAddrs);

  //I need to figure out key ownership
  PHUB_CHECK(numbin % chunkSize == 0);
  int reduceScatterPerMachineKeyCount = numbin;
  int reduceScatterTotalKeyCount = reduceScatterPerMachineKeyCount * num_machines_;
  std::string reduceScatterSupplement = getKeyOwnershipString(num_machines_, reduceScatterPerMachineKeyCount);
  //std::shared_ptr<PHub> createPHubInstance(void *ptr, size_t count, int size, int rank, int instanceId, PHubDataType dataType = PHubDataType::FLOAT, int elementWidth = sizeof(float), std::string scheduleSupplementaryData = "");
  setenv("PLINK_SCHEDULE_TYPE", "reducescatter", 1);
  PHUB_CHECK(pHubBackingBufferForReduceScatter.size() == reduceScatterTotalKeyCount * sizeof(HistogramBinEntry));
  pHubReduceScatter = createPHubInstance(pHubBackingBufferForReduceScatter.data(), reduceScatterTotalKeyCount, num_machines_, rank_, 0, PHubDataType::CUSTOM, sizeof(HistogramBinEntry), reduceScatterSupplement);
  pHubReduceScatter->SetReductionFunction(&PHubHistogramBinEntrySumReducer);

  const int PHUB_ALL_REDUCE_T3_KEY0_SIZE = sizeof(std::tuple<data_size_t, double, double>);
  pHubBackingBufferForAllReduceT3.resize(PHUB_ALL_REDUCE_T3_KEY0_SIZE);
  setenv("PLINK_SCHEDULE_TYPE", "allreduce", 1);
  if (getenv("PHubMaximumCore") != nullptr)
  {
    //sets phubcoreoffset to continue right after max core.
    setenv("PHubCoreOffset", getenv("PHubMaximumCore"), 1);
  }
  pHubAllReduceT3 = createPHubInstance(pHubBackingBufferForAllReduceT3.data(), 1, num_machines_, rank_, 1, PHubDataType::CUSTOM, PHUB_ALL_REDUCE_T3_KEY0_SIZE);
  pHubAllReduceT3->SetReductionFunction(&PHubTuple3Reducer);

  int PHUB_ALL_REDUCE_SPLITINFO_KEY0_SIZE = 2 * SplitInfo::Size(this->config_->max_cat_threshold);
  pHubBackingBufferForAllReduceSplitInfo.resize(PHUB_ALL_REDUCE_SPLITINFO_KEY0_SIZE);
  pHubAllReduceSplitInfo = createPHubInstance(pHubBackingBufferForAllReduceSplitInfo.data(), 2, num_machines_, rank_, 2, PHubDataType::CUSTOM, PHUB_ALL_REDUCE_SPLITINFO_KEY0_SIZE / 2);
  pHubAllReduceSplitInfo->SetReductionFunction(&PHubReducerForSyncUpGlobalBestSplit);

}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::Init(const Dataset *train_data, bool is_constant_hessian)
{
  // initialize SerialTreeLearner
  TREELEARNER_T::Init(train_data, is_constant_hessian);
  //initialize parameter Hub.
  // Get local rank and global machine size
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();
  // allocate buffer for communication
  size_t buffer_size = this->train_data_->NumTotalBin() * sizeof(HistogramBinEntry);

  input_buffer_.resize(buffer_size);
  output_buffer_.resize(buffer_size);

  is_feature_aggregated_.resize(this->num_features_);

  block_start_.resize(num_machines_);
  block_len_.resize(num_machines_);

  buffer_write_start_pos_.resize(this->num_features_);
  buffer_read_start_pos_.resize(this->num_features_);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);

  InitializePHub();
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::ResetConfig(const Config *config)
{
  TREELEARNER_T::ResetConfig(config);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::BeforeTrain()
{
  TREELEARNER_T::BeforeTrain();
  // generate feature partition for current tree
  std::vector<std::vector<int>> feature_distribution(num_machines_, std::vector<int>());
  std::vector<int> num_bins_distributed(num_machines_, 0);

  reduceScatterInnerFid2NodeMapping.resize(this->train_data_->num_total_features());
  for (int i = 0; i < this->train_data_->num_total_features(); ++i)
  {
    int inner_feature_index = this->train_data_->InnerFeatureIndex(i);
    if (inner_feature_index == -1)
    {
      continue;
    }
    if (this->is_feature_used_[inner_feature_index])
    {
      int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));
      feature_distribution[cur_min_machine].push_back(inner_feature_index);
      reduceScatterInnerFid2NodeMapping.at(inner_feature_index) = cur_min_machine;
      auto num_bin = this->train_data_->FeatureNumBin(inner_feature_index);
      if (this->train_data_->FeatureBinMapper(inner_feature_index)->GetDefaultBin() == 0)
      {
        num_bin -= 1;
      }

      num_bins_distributed[cur_min_machine] += num_bin;
    }
    is_feature_aggregated_[inner_feature_index] = false;
  }
  // get local used feature
  for (auto fid : feature_distribution[rank_])
  {
    is_feature_aggregated_[fid] = true;
  }

  // get block start and block len for reduce scatter
  reduce_scatter_size_ = 0;
  for (int i = 0; i < num_machines_; ++i)
  {
    block_len_[i] = 0;
    *(reduceScatterNodeByteCounters.at(i)) = 0;
    for (auto fid : feature_distribution[i])
    {
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0)
      {
        num_bin -= 1;
      }
      block_len_[i] += num_bin * sizeof(HistogramBinEntry);
    }
    reduce_scatter_size_ += block_len_[i];
  }

  // Log::Info("[%d] reduce_scatter_size_ = %d", Network::rank(), reduce_scatter_size_);
  // for (size_t i = 0; i < block_len_.size(); i++)
  // {
  //   Log::Info("[%d] block size = %d. elements = %d. perfectly aligned = %d", Network::rank(), block_len_[i], block_len_[i] / sizeof(HistogramBinEntry), block_len_[i] % sizeof(HistogramBinEntry) == 0);
  // }

  block_start_[0] = 0;
  for (int i = 1; i < num_machines_; ++i)
  {
    block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
  }

  // get buffer_write_start_pos_
  int bin_size = 0;
  for (int i = 0; i < num_machines_; ++i)
  {
    for (auto fid : feature_distribution[i])
    {
      buffer_write_start_pos_[fid] = bin_size;
      auto num_bin = this->train_data_->FeatureNumBin(fid);
      if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0)
      {
        num_bin -= 1;
      }
      bin_size += num_bin * sizeof(HistogramBinEntry);
    }
  }

  // get buffer_read_start_pos_
  bin_size = 0;
  for (auto fid : feature_distribution[rank_])
  {
    buffer_read_start_pos_[fid] = bin_size;
    auto num_bin = this->train_data_->FeatureNumBin(fid);
    if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0)
    {
      num_bin -= 1;
    }
    bin_size += num_bin * sizeof(HistogramBinEntry);
  }

  // sync global data sumup info
  std::tuple<data_size_t, double, double> data(this->smaller_leaf_splits_->num_data_in_leaf(),
                                               this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians());
  int size = sizeof(data);
  std::memcpy(input_buffer_.data(), &data, size);
  // global sumup reduce

  //this is a 20B allreduce. push everything to node 0, key 0.
  Network::Allreduce(input_buffer_.data(), size, sizeof(std::tuple<data_size_t, double, double>), output_buffer_.data(), [](const char *src, char *dst, int type_size, comm_size_t len) {
    comm_size_t used_size = 0;
    const std::tuple<data_size_t, double, double> *p1;
    std::tuple<data_size_t, double, double> *p2;
    while (used_size < len)
    {
      p1 = reinterpret_cast<const std::tuple<data_size_t, double, double> *>(src);
      p2 = reinterpret_cast<std::tuple<data_size_t, double, double> *>(dst);
      std::get<0>(*p2) = std::get<0>(*p2) + std::get<0>(*p1);
      std::get<1>(*p2) = std::get<1>(*p2) + std::get<1>(*p1);
      std::get<2>(*p2) = std::get<2>(*p2) + std::get<2>(*p1);
      src += type_size;
      dst += type_size;
      used_size += type_size;
    }
  });
  // copy back
  std::memcpy(reinterpret_cast<void *>(&data), output_buffer_.data(), size);

  //shadow operation. use this for correctness test.
  //change source direction.

  pHubAllReduceT3->ApplicationSuppliedAddrs.at(0) = &data;
  pHubAllReduceT3->ApplicationSuppliedOutputAddrs.at(0) = &data;
  COMPILER_BARRIER();
  //fine, no race, because syncrhonziation points introduced by work queues.
  pHubAllReduceT3->Reduce();
  // set global sumup info
  this->smaller_leaf_splits_->Init(std::get<1>(data), std::get<2>(data));
  // init global data count in leaf
  global_data_count_in_leaf_[0] = std::get<0>(data);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplits()
{
  TREELEARNER_T::ConstructHistograms(this->is_feature_used_, true);
  // construct local histograms

  //I am skeptical whether OMP will help in this case.
  //#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index)
  {
    if ((!this->is_feature_used_.empty() && this->is_feature_used_[feature_index] == false))
      continue;
    // copy to buffer
    std::memcpy(input_buffer_.data() + buffer_write_start_pos_[feature_index],
                this->smaller_leaf_histogram_array_[feature_index].RawData(),
                this->smaller_leaf_histogram_array_[feature_index].SizeOfHistgram());
    //copy to plink
    auto nodeId = reduceScatterInnerFid2NodeMapping.at(feature_index);
    auto fid2Loc = reduceScatterNodeByteCounters.at(nodeId)->fetch_add(this->smaller_leaf_histogram_array_[feature_index].SizeOfHistgram(), std::memory_order_relaxed) + (char *)reduceScatterNodeStartingAddress.at(nodeId);
    std::memcpy(fid2Loc, this->smaller_leaf_histogram_array_[feature_index].RawData(), this->smaller_leaf_histogram_array_[feature_index].SizeOfHistgram());
    //how do we know where to copy back? we cannot have PLink directly write to output buffer because plink operates at key level.
    //good news is the key assignment makes sure bins belong to the same node are continuous.
  }

  // Reduce scatter for histogram
  Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, sizeof(HistogramBinEntry), block_start_.data(),
                         block_len_.data(), output_buffer_.data(), static_cast<comm_size_t>(output_buffer_.size()), &HistogramBinEntry::SumReducer);
  //for PHub, we need to first figure out keys, and this is very simple
  std::vector<PLinkKey> tasks;
  for (int i = 0; i < num_machines_; i++)
  {
    //check block length agrees
    PLinkKey start = reduceScatterNodeStartingKey.at(i);
    CHECK(block_len_.at(i) == reduceScatterNodeByteCounters.at(i)->load());
    int count = (int)ceil(1.0 * reduceScatterNodeByteCounters.at(i)->load() / sizeof(HistogramBinEntry) / pHubChunkSize );
    for (PLinkKey key = start; key < start + count; key++)
    {
      //plink key supports basic arith,
      tasks.push_back(key);
    }
  }
  pHubReduceScatter->Reduce(tasks);
  //now copy back. simple
  int copyBytes = reduceScatterNodeByteCounters.at(rank_)->load();
  void *srcAddr = reduceScatterNodeStartingAddress.at(rank_);
  std::memcpy(output_buffer_.data() + block_start_.at(rank_), srcAddr, copyBytes);

  this->FindBestSplitsFromHistograms(this->is_feature_used_, true);
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::FindBestSplitsFromHistograms(const std::vector<int8_t> &, bool)
{
  std::vector<SplitInfo> smaller_bests_per_thread(this->num_threads_, SplitInfo());
  std::vector<SplitInfo> larger_bests_per_thread(this->num_threads_, SplitInfo());
  std::vector<int8_t> smaller_node_used_features(this->num_features_, 1);
  std::vector<int8_t> larger_node_used_features(this->num_features_, 1);
  if (this->config_->feature_fraction_bynode < 1.0f)
  {
    smaller_node_used_features = this->GetUsedFeatures(false);
    larger_node_used_features = this->GetUsedFeatures(false);
  }
  OMP_INIT_EX();
#pragma omp parallel for schedule(static)
  for (int feature_index = 0; feature_index < this->num_features_; ++feature_index)
  {
    OMP_LOOP_EX_BEGIN();
    if (!is_feature_aggregated_[feature_index])
      continue;
    const int tid = omp_get_thread_num();
    const int real_feature_index = this->train_data_->RealFeatureIndex(feature_index);
    // restore global histograms from buffer
    this->smaller_leaf_histogram_array_[feature_index].FromMemory(
        output_buffer_.data() + buffer_read_start_pos_[feature_index]);

    this->train_data_->FixHistogram(feature_index,
                                    this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians(),
                                    GetGlobalDataCountInLeaf(this->smaller_leaf_splits_->LeafIndex()),
                                    this->smaller_leaf_histogram_array_[feature_index].RawData());
    SplitInfo smaller_split;
    // find best threshold for smaller child
    this->smaller_leaf_histogram_array_[feature_index].FindBestThreshold(
        this->smaller_leaf_splits_->sum_gradients(),
        this->smaller_leaf_splits_->sum_hessians(),
        GetGlobalDataCountInLeaf(this->smaller_leaf_splits_->LeafIndex()),
        this->smaller_leaf_splits_->min_constraint(),
        this->smaller_leaf_splits_->max_constraint(),
        &smaller_split);
    smaller_split.feature = real_feature_index;
    if (smaller_split > smaller_bests_per_thread[tid] && smaller_node_used_features[feature_index])
    {
      smaller_bests_per_thread[tid] = smaller_split;
    }

    // only root leaf
    if (this->larger_leaf_splits_ == nullptr || this->larger_leaf_splits_->LeafIndex() < 0)
      continue;

    // construct histgroms for large leaf, we init larger leaf as the parent, so we can just subtract the smaller leaf's histograms
    this->larger_leaf_histogram_array_[feature_index].Subtract(
        this->smaller_leaf_histogram_array_[feature_index]);
    SplitInfo larger_split;
    // find best threshold for larger child
    this->larger_leaf_histogram_array_[feature_index].FindBestThreshold(
        this->larger_leaf_splits_->sum_gradients(),
        this->larger_leaf_splits_->sum_hessians(),
        GetGlobalDataCountInLeaf(this->larger_leaf_splits_->LeafIndex()),
        this->larger_leaf_splits_->min_constraint(),
        this->larger_leaf_splits_->max_constraint(),
        &larger_split);
    larger_split.feature = real_feature_index;
    if (larger_split > larger_bests_per_thread[tid] && larger_node_used_features[feature_index])
    {
      larger_bests_per_thread[tid] = larger_split;
    }
    OMP_LOOP_EX_END();
  }
  OMP_THROW_EX();

  auto smaller_best_idx = ArrayArgs<SplitInfo>::ArgMax(smaller_bests_per_thread);
  int leaf = this->smaller_leaf_splits_->LeafIndex();
  this->best_split_per_leaf_[leaf] = smaller_bests_per_thread[smaller_best_idx];

  if (this->larger_leaf_splits_ != nullptr && this->larger_leaf_splits_->LeafIndex() >= 0)
  {
    leaf = this->larger_leaf_splits_->LeafIndex();
    auto larger_best_idx = ArrayArgs<SplitInfo>::ArgMax(larger_bests_per_thread);
    this->best_split_per_leaf_[leaf] = larger_bests_per_thread[larger_best_idx];
  }

  SplitInfo smaller_best_split, larger_best_split;
  smaller_best_split = this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()];
  // find local best split for larger leaf
  if (this->larger_leaf_splits_->LeafIndex() >= 0)
  {
    larger_best_split = this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()];
  }

  // sync global best info
  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold, pHubAllReduceSplitInfo);

  // set best split
  this->best_split_per_leaf_[this->smaller_leaf_splits_->LeafIndex()] = smaller_best_split;
  if (this->larger_leaf_splits_->LeafIndex() >= 0)
  {
    this->best_split_per_leaf_[this->larger_leaf_splits_->LeafIndex()] = larger_best_split;
  }
}

template <typename TREELEARNER_T>
void DataParallelTreeLearner<TREELEARNER_T>::Split(Tree *tree, int best_Leaf, int *left_leaf, int *right_leaf)
{
  TREELEARNER_T::Split(tree, best_Leaf, left_leaf, right_leaf);
  const SplitInfo &best_split_info = this->best_split_per_leaf_[best_Leaf];
  // need update global number of data in leaf
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
}

// instantiate template classes, otherwise linker cannot find the code
template class DataParallelTreeLearner<GPUTreeLearner>;
template class DataParallelTreeLearner<SerialTreeLearner>;

} // namespace LightGBM
