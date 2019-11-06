/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#include <cstring>
#include <tuple>
#include <vector>
#include <math.h>

#include "parallel_tree_learner.h"
#include "benchmark_tree_learner.h"
#include <LightGBM/utils/common.h>

namespace LightGBM
{

template <typename TREELEARNER_T>
BenchmarkParallelTreeLearner<TREELEARNER_T>::BenchmarkParallelTreeLearner(const Config *config)
    : TREELEARNER_T(config)
{
}

template <typename TREELEARNER_T>
BenchmarkParallelTreeLearner<TREELEARNER_T>::~BenchmarkParallelTreeLearner()
{
  if (pHubAllReduceT3 != nullptr)
  {
    pHubAllReduceT3->FastTerminate();
  }
  if (pHubAllReduceSplitInfo != nullptr)
  {
    pHubAllReduceSplitInfo->FastTerminate();
  }
  if (pHubReduceScatter != nullptr)
  {
    pHubReduceScatter->FastTerminate();
  }
}

template <typename TREELEARNER_T>
void BenchmarkParallelTreeLearner<TREELEARNER_T>::InitializePHub()
{
  rank_ = Network::rank();
  num_machines_ = Network::num_machines();

  // allocate buffer for communication
  auto chunkSize = atoi(pHubGetMandatoryEnvironmemtVariable("PHubChunkElementSize").c_str());
  pHubChunkSize = chunkSize;
  size_t numbin = RoundUp(this->train_data_->NumTotalBin(), chunkSize);
  fprintf(stderr, "benchmarked [%d] numbin = %d, adjusted = %d. chunk = %d\n", rank_, (int)this->train_data_->NumTotalBin(), (int)numbin, (int)chunkSize);
  size_t buffer_size = numbin * sizeof(HistogramBinEntry);
  reduceScatterPerNodeBufferSize = buffer_size;
  auto total_buffer_size = buffer_size * num_machines_;

  pHubBackingBufferForReduceScatter.resize(total_buffer_size);
  reduceScatterNodeStartingAddress.resize(num_machines_);
  reduceScatterNodeStartingKey.resize(num_machines_);
  reduceScatterBlockLenAccSum.resize(num_machines_);
  auto commBackend = std::string(std::getenv("BENCHMARK_PREFERRED_BACKEND") == nullptr ? "" : std::getenv("BENCHMARK_PREFERRED_BACKEND"));
  if (commBackend == "PHUB")
  {
    for (int i = 0; i < num_machines_; i++)
    {
      reduceScatterNodeByteCounters.push_back(std::make_unique<std::atomic<int>>(0));
      reduceScatterNodeStartingAddress.at(i) = pHubBackingBufferForReduceScatter.data() + i * buffer_size;
      reduceScatterNodeStartingKey.at(i) = i * (numbin / chunkSize);
      //reduceScatterNodeFidOrder.push_back(std::vector<int>());
      //reduceScatterNodeFidOrder.reserve(numbin);
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
    //here, getKeyOwnershipString is speaking PHub key, which is a collection of bins.
    std::string reduceScatterSupplement = getKeyOwnershipString(num_machines_, reduceScatterPerMachineKeyCount / chunkSize);
    setenv("PLINK_SCHEDULE_TYPE", "reducescatter", 1);
    PHUB_CHECK(pHubBackingBufferForReduceScatter.size() == reduceScatterTotalKeyCount * sizeof(HistogramBinEntry));
    pHubReduceScatter = createPHubInstance(pHubBackingBufferForReduceScatter.data(), reduceScatterTotalKeyCount, num_machines_, rank_, 0, PHubDataType::CUSTOM, sizeof(HistogramBinEntry), reduceScatterSupplement);
    PHUB_CHECK(pHubReduceScatter->keySizes.size() == num_machines_ * numbin / chunkSize);
    pHubReduceScatter->SetReductionFunction(&PHubHistogramBinEntrySumReducer);

    /*
    const int PHUB_ALL_REDUCE_T3_KEY0_SIZE = sizeof(std::tuple<data_size_t, double, double>);
    pHubBackingBufferForAllReduceT3.resize(PHUB_ALL_REDUCE_T3_KEY0_SIZE);
    setenv("PLINK_SCHEDULE_TYPE", "allreduce", 1);
    int reduceScatterCores = 1;
    if (getenv("PHubMaximumCore") != nullptr)
    {
      reduceScatterCores = atoi(getenv("PHubMaximumCore"));
      //sets phubcoreoffset to continue right after max core.
      setenv("PHubCoreOffset", getenv("PHubMaximumCore"), 1);
      setenv("PHubMaximumCore", "1", 1);
    }
    
    setenv("PHubChunkElementSize", "1", 1);
    pHubAllReduceT3 = createPHubInstance(pHubBackingBufferForAllReduceT3.data(), 1, num_machines_, rank_, 1, PHubDataType::CUSTOM, PHUB_ALL_REDUCE_T3_KEY0_SIZE);
    pHubAllReduceT3->SetReductionFunction(&PHubTuple3Reducer);
    PHUB_CHECK(pHubAllReduceT3->keySizes.size() == 1 && (size_t)pHubAllReduceT3->keySizes.at(0) == pHubBackingBufferForAllReduceT3.size());
    
    if (getenv("PHubMaximumCore") != nullptr)
    {
      //sets phubcoreoffset to continue right after max core.
      auto reqCore = reduceScatterCores + 1; //this is for reduce scatter, plus the t3 phub
      setenv("PHubCoreOffset", std::to_string(reqCore).c_str(), 1);
    }
    int PHUB_ALL_REDUCE_SPLITINFO_KEY0_SIZE = 2 * SplitInfo::Size(this->config_->max_cat_threshold);
    pHubBackingBufferForAllReduceSplitInfo.resize(PHUB_ALL_REDUCE_SPLITINFO_KEY0_SIZE);
    setenv("PHubChunkElementSize", "2", 1);
    pHubAllReduceSplitInfo = createPHubInstance(pHubBackingBufferForAllReduceSplitInfo.data(), 2, num_machines_, rank_, 2, PHubDataType::CUSTOM, PHUB_ALL_REDUCE_SPLITINFO_KEY0_SIZE / 2);
    PHUB_CHECK(pHubAllReduceSplitInfo->keySizes.size() == 1 && (size_t)pHubAllReduceSplitInfo->keySizes.at(0) == pHubBackingBufferForAllReduceSplitInfo.size());

    pHubAllReduceSplitInfo->SetReductionFunction(&PHubReducerForSyncUpGlobalBestSplit);
    //both write to input_buffer.
    pHubAllReduceSplitInfo->ApplicationSuppliedOutputAddrs.at(0) = input_buffer_.data(); //pHubBackingBufferForAllReduceSplitInfo.data();
    pHubAllReduceSplitInfo->ApplicationSuppliedAddrs.at(0) = input_buffer_.data();
    */
  }
}

template <typename TREELEARNER_T>
void BenchmarkParallelTreeLearner<TREELEARNER_T>::Init(const Dataset *train_data, bool is_constant_hessian)
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

  auto commBackend = std::string(std::getenv("BENCHMARK_PREFERRED_BACKEND") == nullptr ? "" : std::getenv("BENCHMARK_PREFERRED_BACKEND"));

  InitializePHub();
  //}

  //reset real impl to use same size as PHub
  input_buffer_.resize(pHubBackingBufferForReduceScatter.size());
  output_buffer_.resize(pHubBackingBufferForReduceScatter.size());

  //now, equally partition this to reduce scatter.
  reduce_scatter_size_ = pHubBackingBufferForReduceScatter.size();
  comm_size_t perNode = reduce_scatter_size_ / num_machines_;
  for (int i = 0; i < num_machines_; i++)
  {
    block_len_.at(i) = perNode;
    block_start_.at(i) = i == 0 ? 0 : block_start_.at(i - 1) + perNode;
  }

  //get communication backend.
  if (commBackend == "" || commBackend == "DEFAULT")
  {
    fprintf(stderr, "[%d] Default benchmark activated. total = %f MB. perNode = %f MB\n", Network::rank(), reduce_scatter_size_ / 1024.0 / 1024.0, perNode / 1024.0 / 1024.0);
    benchmarkCommBackend = BenchmarkPreferredBackend::DEFAULT;
  }
  else if (commBackend == "PHUB")
  {
    benchmarkCommBackend = BenchmarkPreferredBackend::PHUB;
  }
  else
  {
    PHUB_CHECK(false);
  }
}

template <typename TREELEARNER_T>
void BenchmarkParallelTreeLearner<TREELEARNER_T>::ResetConfig(const Config *config)
{
  TREELEARNER_T::ResetConfig(config);
  global_data_count_in_leaf_.resize(this->config_->num_leaves);
}

template <typename TREELEARNER_T>
void BenchmarkParallelTreeLearner<TREELEARNER_T>::BeforeTrain()
{
  //EASY_FUNCTION(profiler::colors::Red50);
  //TREELEARNER_T::BeforeTrain();
  // generate feature partition for current tree
  // std::vector<std::vector<int>> feature_distribution(num_machines_, std::vector<int>());
  // std::vector<int> num_bins_distributed(num_machines_, 0);

  // reduceScatterInnerFid2NodeMapping.resize(this->train_data_->num_total_features());
  // for (int i = 0; i < this->train_data_->num_total_features(); ++i)
  // {
  //   int inner_feature_index = this->train_data_->InnerFeatureIndex(i);
  //   if (inner_feature_index == -1)
  //   {
  //     continue;
  //   }
  //   if (this->is_feature_used_[inner_feature_index])
  //   {
  //     int cur_min_machine = static_cast<int>(ArrayArgs<int>::ArgMin(num_bins_distributed));
  //     feature_distribution[cur_min_machine].push_back(inner_feature_index);
  //     //unfortunately inneridx is orderless.
  //     //fprintf(stderr, "[%d] instatiate order: f_d[%d] . append(%d)\n", cur_min_machine, inner_feature_index);
  //     reduceScatterInnerFid2NodeMapping.at(inner_feature_index) = cur_min_machine;
  //     auto num_bin = this->train_data_->FeatureNumBin(inner_feature_index);
  //     if (this->train_data_->FeatureBinMapper(inner_feature_index)->GetDefaultBin() == 0)
  //     {
  //       num_bin -= 1;
  //     }

  //     num_bins_distributed[cur_min_machine] += num_bin;
  //   }
  //   is_feature_aggregated_[inner_feature_index] = false;
  // }

  // // get local used feature
  // for (auto fid : feature_distribution[rank_])
  // {
  //   is_feature_aggregated_[fid] = true;
  // }

  // // get block start and block len for reduce scatter
  // reduce_scatter_size_ = 0;
  // for (int i = 0; i < num_machines_; ++i)
  // {
  //   block_len_[i] = 0;
  //   *(reduceScatterNodeByteCounters.at(i)) = 0;
  //   for (auto fid : feature_distribution[i])
  //   {
  //     auto num_bin = this->train_data_->FeatureNumBin(fid);
  //     if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0)
  //     {
  //       num_bin -= 1;
  //     }
  //     block_len_[i] += num_bin * sizeof(HistogramBinEntry);
  //   }
  //   reduceScatterBlockLenAccSum.at(i) = i == 0 ? block_len_[i] : block_len_[i] + reduceScatterBlockLenAccSum.at(i - 1);
  //   reduce_scatter_size_ += block_len_[i];
  // }

  // //fprintf(stderr, "[%d] reduce_scatter size = %d\n", rank_, reduce_scatter_size_);
  // // Log::Info("[%d] reduce_scatter_size_ = %d", Network::rank(), reduce_scatter_size_);
  // // for (size_t i = 0; i < block_len_.size(); i++)
  // // {
  // //   Log::Info("[%d] block size = %d. elements = %d. perfectly aligned = %d", Network::rank(), block_len_[i], block_len_[i] / sizeof(HistogramBinEntry), block_len_[i] % sizeof(HistogramBinEntry) == 0);
  // // }

  // block_start_[0] = 0;
  // for (int i = 1; i < num_machines_; ++i)
  // {
  //   block_start_[i] = block_start_[i - 1] + block_len_[i - 1];
  // }

  // // get buffer_write_start_pos_
  // int bin_size = 0;
  // for (int i = 0; i < num_machines_; ++i)
  // {
  //   for (auto fid : feature_distribution[i])
  //   {
  //     buffer_write_start_pos_[fid] = bin_size;
  //     //fprintf(stderr, "[%d] fid = %d, target = %d, start offset = %d\n", rank_, fid, i, bin_size);
  //     auto num_bin = this->train_data_->FeatureNumBin(fid);
  //     if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0)
  //     {
  //       num_bin -= 1;
  //     }
  //     bin_size += num_bin * sizeof(HistogramBinEntry);
  //   }
  // }

  // // get buffer_read_start_pos_
  // bin_size = 0;
  // for (auto fid : feature_distribution[rank_])
  // {
  //   buffer_read_start_pos_[fid] = bin_size;
  //   auto num_bin = this->train_data_->FeatureNumBin(fid);
  //   if (this->train_data_->FeatureBinMapper(fid)->GetDefaultBin() == 0)
  //   {
  //     num_bin -= 1;
  //   }
  //   bin_size += num_bin * sizeof(HistogramBinEntry);
  // }
  // sync global data sumup info
  std::tuple<data_size_t, double, double> data(this->smaller_leaf_splits_->num_data_in_leaf(),
                                               this->smaller_leaf_splits_->sum_gradients(), this->smaller_leaf_splits_->sum_hessians());
  //shadow operation. use this for correctness test.
  //change source direction.
  switch (benchmarkCommBackend)
  {
  // case BenchmarkPreferredBackend::PHUB:
  // {
  //   pHubAllReduceT3->ApplicationSuppliedAddrs.at(0) = &data;       //&data1;
  //   pHubAllReduceT3->ApplicationSuppliedOutputAddrs.at(0) = &data; //&data1;
  //   COMPILER_BARRIER();
  //   //fine, no race, because syncrhonziation points introduced by work queues.
  //   EASY_BLOCK("PHub T3 AllReduce");
  //   pHubAllReduceT3->Reduce();
  //   EASY_END_BLOCK;
  //   break;
  // }
  case BenchmarkPreferredBackend::DEFAULT:
  {
    int size = sizeof(data);
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
    break;
  }
  default:
    break;
  }

  // set global sumup info
  //this->smaller_leaf_splits_->Init(std::get<1>(data), std::get<2>(data));
  // init global data count in leaf
  //global_data_count_in_leaf_[0] = std::get<0>(data);
}

template <typename TREELEARNER_T>
void BenchmarkParallelTreeLearner<TREELEARNER_T>::FindBestSplits()
{
  //fprintf(stderr, "[%d]benchmarked tree learner . FindBestSplits\n", Network::rank());

  EASY_FUNCTION(profiler::colors::Magenta);

  // switch (benchmarkCommBackend)
  // {
  // case BenchmarkPreferredBackend::DEFAULT:
  // {
  //   EASY_BLOCK("Default_ReduceScatter");
  //   Network::ReduceScatter(input_buffer_.data(), reduce_scatter_size_, sizeof(HistogramBinEntry), block_start_.data(),
  //                          block_len_.data(), output_buffer_.data(), static_cast<comm_size_t>(output_buffer_.size()), &HistogramBinEntry::SumReducer);
  //   EASY_END_BLOCK;
  //   break;
  // }
  // case BenchmarkPreferredBackend::PHUB:
  // {
  //   EASY_BLOCK("PHub_ReduceScatter");
  //   pHubReduceScatter->Reduce();
  //   EASY_END_BLOCK;
  //   break;
  // }
  // default:
  //   break;
  // }

  this->FindBestSplitsFromHistograms(this->is_feature_used_, true);
}

template <typename TREELEARNER_T>
void BenchmarkParallelTreeLearner<TREELEARNER_T>::FindBestSplitsFromHistograms(const std::vector<int8_t> &, bool)
{
  //return;
  //fprintf(stderr, "[%d]benchmarked tree learner . FindBestSplitsFromHistograms\n", Network::rank());
  SplitInfo smaller_best_split, larger_best_split;
  //smaller_best_split = SplitInfo();
  // find local best split for larger leaf
  //larger_best_split = SplitInfo();
  //fprintf(stderr, "[%d]benchmarked tree learner . FindBestSplitsFromHistograms.418\n", Network::rank());
  //all ignored.
  // sync global best info
  //switch (benchmarkCommBackend)
  //{
  //case BenchmarkPreferredBackend::DEFAULT:
  //{
  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold);
  //  break;
  //}
  //case BenchmarkPreferredBackend::PHUB:
  //{
  //  SyncUpGlobalBestSplit(input_buffer_.data(), input_buffer_.data(), &smaller_best_split, &larger_best_split, this->config_->max_cat_threshold, pHubAllReduceSplitInfo);
  //  break;
  //}
  //default:
  //   break;
  //}
}

template <typename TREELEARNER_T>
void BenchmarkParallelTreeLearner<TREELEARNER_T>::Split(Tree *tree, int best_Leaf, int *left_leaf, int *right_leaf)
{
  TREELEARNER_T::Split(tree, best_Leaf, left_leaf, right_leaf);
  const SplitInfo &best_split_info = this->best_split_per_leaf_[best_Leaf];
  // need update global number of data in leaf
  global_data_count_in_leaf_[*left_leaf] = best_split_info.left_count;
  global_data_count_in_leaf_[*right_leaf] = best_split_info.right_count;
}

// instantiate template classes, otherwise linker cannot find the code
template class BenchmarkParallelTreeLearner<BenchmarkTreeLearner>;

} // namespace LightGBM
