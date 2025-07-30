/* This code and any associated documentation is provided "as is"

Copyright 2025 Munich Quantum Software Stack Project

Licensed under the Apache License, Version 2.0 with LLVM Exceptions (the
"License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://github.com/Munich-Quantum-Software-Stack/MQSS-CUDAQ-Adapter/tree/develop?tab=readme-ov-file#

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-------------------------------------------------------------------------
  author Martin Letras
  @date   November 2024
  @version 1.0
  @ brief

  Enum that defines the status of a job returned by the MQSS via the MQP.

*******************************************************************************
* This source code and the accompanying materials are made available under    *
* the terms of the Apache License 2.0 which accompanies this distribution.    *
******************************************************************************/

#ifndef MQSSJOBSTATUS_H
#define MQSSJOBSTATUS_H

// Define an enumeration for JobStatus
namespace cudaq::mqss {
enum class JobStatus { PENDING, WAITING, FAILED, COMPLETED, CANCELLED };

const std::string jobStatusToString(JobStatus jobStatus) {
  switch (jobStatus) {
  case JobStatus::PENDING:
    return "PENDING";
  case JobStatus::WAITING:
    return "WAITING";
  case JobStatus::FAILED:
    return "FAILED";
  case JobStatus::COMPLETED:
    return "COMPLETED";
  case JobStatus::CANCELLED:
    return "CANCELLED";
  }
  return "UNKNOWN";
}
} // namespace cudaq::mqss
#endif // MQSS_JOBSTATUS_H
