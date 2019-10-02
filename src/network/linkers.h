/*!
 * Copyright (c) 2016 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for license information.
 */
#ifndef LIGHTGBM_NETWORK_LINKERS_H_
#define LIGHTGBM_NETWORK_LINKERS_H_

#include <LightGBM/config.h>
#include <LightGBM/meta.h>
#include <LightGBM/network.h>
#include <LightGBM/utils/common.h>

#include <string>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <memory>
#include <thread>
#include <vector>

#ifdef USE_SOCKET
#include "socket_wrapper.hpp"
#endif

#ifdef USE_MPI
#include <mpi.h>
#define MPI_SAFE_CALL(mpi_return) CHECK((mpi_return) == MPI_SUCCESS)
#endif

namespace LightGBM
{

class Timer
{

protected:
  std::chrono::time_point<std::chrono::high_resolution_clock>
      start_;

public:
  Timer()
  {
    start();
  }

  void start()
  {
    start_ = std::chrono::high_resolution_clock::now();
  }

  long ns() const
  {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::nanoseconds(now - start_).count();
  }

  double s() const
  {
    auto now = std::chrono::high_resolution_clock::now();
    return 1.0 * std::chrono::nanoseconds(now - start_).count() * 1e-9;
  }
};

/*!
* \brief An network basic communication warpper.
* Will warp low level communication methods, e.g. mpi, socket and so on.
* This class will wrap all linkers to other machines if needs
*/
class Linkers
{
public:
  Linkers()
  {
    is_init_ = false;
  }
  inline size_t GetInferredBytesTransferred()
  {
    return InferredTranferredBytes;
  }
  /*!
  * \brief Constructor
  * \param config Config of network settings
  */
  explicit Linkers(Config config);
  /*!
  * \brief Destructor
  */
  ~Linkers();
  /*!
  * \brief Recv data, blocking
  * \param rank Which rank will send data to local machine
  * \param data Pointer of receive data
  * \prama len Recv size, will block until recive len size of data
  */
  inline void Recv(int rank, char *data, int len);

  inline void Recv(int rank, char *data, int64_t len);

  /*!
  * \brief Send data, blocking
  * \param rank Which rank local machine will send to
  * \param data Pointer of send data
  * \prama len Send size
  */
  inline void Send(int rank, char *data, int len);

  inline void Send(int rank, char *data, int64_t len);
  /*!
  * \brief Send and Recv at same time, blocking
  * \param send_rank
  * \param send_data
  * \prama send_len
  * \param recv_rank
  * \param recv_data
  * \prama recv_len
  */
  inline void SendRecv(int send_rank, char *send_data, int send_len,
                       int recv_rank, char *recv_data, int recv_len);

  inline void SendRecv(int send_rank, char *send_data, int64_t send_len,
                       int recv_rank, char *recv_data, int64_t recv_len);
  /*!
  * \brief Get rank of local machine
  */
  inline int rank();
  /*!
  * \brief Get total number of machines
  */
  inline int num_machines();
  /*!
  * \brief Get Bruck map of this network
  */
  inline const BruckMap &bruck_map();
  /*!
  * \brief Get Recursive Halving map of this network
  */
  inline const RecursiveHalvingMap &recursive_halving_map();
  //perf tracker can be exposed no problem.
  std::size_t InferredTranferredBytes;
  double NetworkSendTime;
  double NetworkRecvTime;
  double NetworkSendRecvTime;
#ifdef USE_SOCKET
  /*!
  * \brief Bind local listen to port
  * \param port Local listen port
  */
  void TryBind(int port);
  /*!
  * \brief Set socket to rank
  * \param rank
  * \param socket
  */
  void SetLinker(int rank, const TcpSocket &socket);
  /*!
  * \brief Thread for listening
  * \param incoming_cnt Number of incoming machines
  */
  void ListenThread(int incoming_cnt);
  /*!
  * \brief Construct network topo
  */
  void Construct();
  /*!
  * \brief Parser machines information from file
  * \param machines
  * \param filename
  */
  void ParseMachineList(const std::string &machines, const std::string &filename);
  /*!
  * \brief Check one linker is connected or not
  * \param rank
  * \return True if linker is connected
  */
  bool CheckLinker(int rank);
  /*!
  * \brief Print connented linkers
  */
  void PrintLinkers();

#endif // USE_SOCKET


private:
  /*! \brief Rank of local machine */
  int rank_;
  /*! \brief Total number machines */
  int num_machines_;
  /*! \brief Bruck map */
  BruckMap bruck_map_;
  /*! \brief Recursive Halving map */
  RecursiveHalvingMap recursive_halving_map_;

  std::chrono::duration<double, std::milli> network_time_;

  bool is_init_;

#ifdef USE_SOCKET
  /*! \brief use to store client ips */
  std::vector<std::string> client_ips_;
  /*! \brief use to store client ports */
  std::vector<int> client_ports_;
  /*! \brief time out for sockets, in minutes */
  int socket_timeout_;
  /*! \brief Local listen ports */
  int local_listen_port_;
  /*! \brief Linkers */
  std::vector<std::unique_ptr<TcpSocket>> linkers_;
  /*! \brief Local socket listener */
  std::unique_ptr<TcpSocket> listener_;
#endif // USE_SOCKET
};

inline int Linkers::rank()
{
  return rank_;
}

inline int Linkers::num_machines()
{
  return num_machines_;
}

inline const BruckMap &Linkers::bruck_map()
{
  return bruck_map_;
}

inline const RecursiveHalvingMap &Linkers::recursive_halving_map()
{
  return recursive_halving_map_;
}

inline void Linkers::Recv(int rank, char *data, int64_t len)
{
  int64_t used = 0;
  do
  {
    int cur_size = static_cast<int>(std::min<int64_t>(len - used, INT32_MAX));
    Recv(rank, data + used, cur_size);
    used += cur_size;
  } while (used < len);
}

inline void Linkers::Send(int rank, char *data, int64_t len)
{
  int64_t used = 0;
  do
  {
    int cur_size = static_cast<int>(std::min<int64_t>(len - used, INT32_MAX));
    Send(rank, data + used, cur_size);
    used += cur_size;
  } while (used < len);
}

inline void Linkers::SendRecv(int send_rank, char *send_data, int64_t send_len,
                              int recv_rank, char *recv_data, int64_t recv_len)
{
  std::thread send_worker(
      [this, send_rank, send_data, send_len]() {
        Send(send_rank, send_data, send_len);
      });
  Recv(recv_rank, recv_data, recv_len);
  send_worker.join();
  // wait for send complete
}

#ifdef USE_SOCKET

inline void Linkers::Recv(int rank, char *data, int len)
{
  Timer t;
  int recv_cnt = 0;
  while (recv_cnt < len)
  {
    recv_cnt += linkers_[rank]->Recv(data + recv_cnt,
                                     // len - recv_cnt
                                     std::min(len - recv_cnt, SocketConfig::kMaxReceiveSize));
  }
  InferredTranferredBytes += len;
  NetworkRecvTime += t.s();
}

inline void Linkers::Send(int rank, char *data, int len)
{
  Timer t;
  if (len <= 0)
  {
    return;
  }
  int send_cnt = 0;
  while (send_cnt < len)
  {
    send_cnt += linkers_[rank]->Send(data + send_cnt, len - send_cnt);
  }
  InferredTranferredBytes += len;
  NetworkSendTime += t.s();
}

inline void Linkers::SendRecv(int send_rank, char *send_data, int send_len,
                              int recv_rank, char *recv_data, int recv_len)
{
  Timer t;
  if (send_len < SocketConfig::kSocketBufferSize)
  {
    // if buffer is enough, send will non-blocking
    Send(send_rank, send_data, send_len);
    Recv(recv_rank, recv_data, recv_len);
  }
  else
  {
    // if buffer is not enough, use another thread to send, since send will be blocking
    std::thread send_worker(
        [this, send_rank, send_data, send_len]() {
          Send(send_rank, send_data, send_len);
        });
    Recv(recv_rank, recv_data, recv_len);
    send_worker.join();
  }
  NetworkSendRecvTime += t.s();
}

#endif // USE_SOCKET

#ifdef USE_MPI

inline void Linkers::Recv(int rank, char *data, int len)
{
  Timer t;
  MPI_Status status;
  int read_cnt = 0;
  while (read_cnt < len)
  {
    MPI_SAFE_CALL(MPI_Recv(data + read_cnt, len - read_cnt, MPI_BYTE, rank, MPI_ANY_TAG, MPI_COMM_WORLD, &status));
    int cur_cnt;
    MPI_SAFE_CALL(MPI_Get_count(&status, MPI_BYTE, &cur_cnt));
    read_cnt += cur_cnt;
  }
  InferredTranferredBytes += len;
  NetworkRecvTime += t.s();
}

inline void Linkers::Send(int rank, char *data, int len)
{
  Timer t;
  if (len <= 0)
  {
    return;
  }
  MPI_Status status;
  MPI_Request send_request;
  MPI_SAFE_CALL(MPI_Isend(data, len, MPI_BYTE, rank, 0, MPI_COMM_WORLD, &send_request));
  MPI_SAFE_CALL(MPI_Wait(&send_request, &status));
  InferredTranferredBytes += len;
  NetworkSendTime += t.s();
}

inline void Linkers::SendRecv(int send_rank, char *send_data, int send_len,
                              int recv_rank, char *recv_data, int recv_len)
{
  Timer t;
  MPI_Request send_request;
  // send first, non-blocking
  MPI_SAFE_CALL(MPI_Isend(send_data, send_len, MPI_BYTE, send_rank, 0, MPI_COMM_WORLD, &send_request));
  // then receive, blocking
  MPI_Status status;
  int read_cnt = 0;
  while (read_cnt < recv_len)
  {
    MPI_SAFE_CALL(MPI_Recv(recv_data + read_cnt, recv_len - read_cnt, MPI_BYTE, recv_rank, 0, MPI_COMM_WORLD, &status));
    int cur_cnt;
    MPI_SAFE_CALL(MPI_Get_count(&status, MPI_BYTE, &cur_cnt));
    read_cnt += cur_cnt;
  }
  // wait for send complete
  MPI_SAFE_CALL(MPI_Wait(&send_request, &status));
  InferredTranferredBytes += send_len + recv_len;
  NetworkSendRecvTime += t.s();
}

#endif // USE_MPI
} // namespace LightGBM
#endif // LightGBM_NETWORK_LINKERS_H_
