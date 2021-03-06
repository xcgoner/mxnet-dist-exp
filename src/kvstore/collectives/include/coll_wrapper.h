/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/**
 * Copyright (c) 2018 by Contributors
 */

#ifndef MXNET_KVSTORE_COLLECTIVES_INCLUDE_COLL_WRAPPER_H_
#define MXNET_KVSTORE_COLLECTIVES_INCLUDE_COLL_WRAPPER_H_

#if MXNET_USE_ALLREDUCE_DIST_KVSTORE

#include <mpi.h>

#include "mxnet/ndarray.h"
#include "mxnet/base.h"
#include "mpi_message.pb.h"

template<typename DType>
MPI_Datatype MPI_Data_Type_Cast(void);

template<>
MPI_Datatype MPI_Data_Type_Cast<int>(void) {
  return MPI_INT;
}

template<>
MPI_Datatype MPI_Data_Type_Cast<float>(void) {
  return MPI_FLOAT;
}

template<>
MPI_Datatype MPI_Data_Type_Cast<double>(void) {
  return MPI_DOUBLE;
}

template <class xpu, class DType>
struct COLL_Wrapper {
  static int Broadcast(mxnet::NDArray *input_array,
                       int root_rank) {
    return 0; }

  static int AllReduce(mxnet::NDArray *input_array,
                       mxnet::NDArray *output_array) {
    return 0; }
};

// CPU Implementation
template <class DType>
struct COLL_Wrapper<mxnet::cpu, DType> {
  static int Broadcast(mxnet::NDArray *input_array,
                       int root_rank) {
    DType *buf = reinterpret_cast<DType *>(input_array->data().dptr<DType>());
    unsigned int count = input_array->data().Size();
    int ret = MPI_Bcast(buf, count, MPI_Data_Type_Cast<DType>(), root_rank, MPI_COMM_WORLD);
    return ret;
  }

  static int AllReduce(mxnet::NDArray *input_array,
                       mxnet::NDArray *output_array) {
    DType *send_buf = reinterpret_cast<DType *>(input_array->data().dptr<DType>());
    DType *recv_buf = reinterpret_cast<DType *>(output_array->data().dptr<DType>());
    unsigned int count = input_array->data().Size();
    int ret;
    assert(input_array->data().Size() == output_array->data().Size());

    if (send_buf != recv_buf) {
      ret = MPI_Allreduce(reinterpret_cast<const void *>(send_buf),
                          reinterpret_cast<void *>(recv_buf),
                          count, MPI_Data_Type_Cast<DType>(), MPI_SUM, MPI_COMM_WORLD);
    } else {
      ret = MPI_Allreduce(MPI_IN_PLACE, reinterpret_cast<void *>(recv_buf),
                         count, MPI_Data_Type_Cast<DType>(), MPI_SUM, MPI_COMM_WORLD);
    }
    return ret;
  }
};

// GPU Implementation
template <class DType>
struct COLL_Wrapper<mxnet::gpu, DType> {
  // static int Broadcast(mxnet::NDArray *input_array,
  //                      int root_rank) {
  //   // TODO(zhouhaiy): implement gpu broadcast
  //   LOG(FATAL) << "Collective For GPU version has not been implemented.";
  //   return -1;
  // }

  // static int AllReduce(mxnet::NDArray *input_array,
  //                      mxnet::NDArray *output_array) {
  //   // TODO(zhouhaiy): implement gpu all reduce
  //   LOG(FATAL) << "Collective For GPU version has not been implemented.";
  //   return -1;
  // }
  static int Broadcast(mxnet::NDArray *input_array,
                       int root_rank) {
    // manually copy to cpu
    // TODO: simply use MPI CUDA-aware API (change makefile and dependency)
    mxnet::NDArray cpu_buf = mxnet::NDArray(input_array->shape(), mxnet::Context::CPU(), true, input_array->dtype());
    CopyFromTo(*input_array, &cpu_buf);
    DType *buf = reinterpret_cast<DType *>(cpu_buf.data().dptr<DType>());
    unsigned int count = input_array->data().Size();
    int ret = MPI_Bcast(buf, count, MPI_Data_Type_Cast<DType>(), root_rank, MPI_COMM_WORLD);
    assert(ret == MPI_SUCCESS);
    CopyFromTo(cpu_buf, input_array);
    return ret;
  }

  static int AllReduce(mxnet::NDArray *input_array,
                       mxnet::NDArray *output_array) {
    unsigned int count = input_array->data().Size();
    int ret;
    assert(input_array->data().Size() == output_array->data().Size());

    mxnet::NDArray cpu_buf = mxnet::NDArray(input_array->shape(), mxnet::Context::CPU(), true, input_array->dtype());
    CopyFromTo(*input_array, &cpu_buf);
    DType *send_buf = reinterpret_cast<DType *>(cpu_buf.data().dptr<DType>());
    ret = MPI_Allreduce(MPI_IN_PLACE, reinterpret_cast<void *>(send_buf),
                        count, MPI_Data_Type_Cast<DType>(), MPI_SUM, MPI_COMM_WORLD);
    CopyFromTo(cpu_buf, output_array);
    return ret;
  }
};

#endif
#endif  // MXNET_KVSTORE_COLLECTIVES_INCLUDE_COLL_WRAPPER_H_
