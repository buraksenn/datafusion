<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements. See the NOTICE file
distributed with this work for additional information
regarding copyright ownership. The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied. See the License for the
specific language governing permissions and limitations
under the License.
-->

# Physical Plan Serde Follow-up Fixes

This branch is stacked on `24624-file-scan-proto-destructure` and contains
serialization fixes intentionally separated from that issue-focused change.

## Changes

- Serialize file-source virtual columns so plans using `file_row_index()` can be
  decoded after a protobuf round trip.
- Serialize the full `CastExpr` target field so output names, nullability, and
  metadata survive that round trip. Legacy payloads without the field retain
  type-only cast behavior.
- Reconstruct file and partition schemas by position rather than by name, which
  preserves schemas where a file column and partition column share a name.
- Reject `FileScanConfig` values with a zero batch size during encoding instead
  of producing a payload that the decoder rejects.

## Regression Coverage

- A Parquet `file_row_index()` physical-plan round trip.
- Cast target-field metadata and malformed target-type handling.
- Name-colliding file and partition columns.
- Encode-side zero batch-size validation.
