/*
 *  Licensed to the Apache Software Foundation (ASF) under one
 *  or more contributor license agreements.  See the NOTICE file
 *  distributed with this work for additional information
 *  regarding copyright ownership.  The ASF licenses this file
 *  to you under the Apache License, Version 2.0 (the
 *  "License"); you may not use this file except in compliance
 *  with the License.  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an
 *  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 *  KIND, either express or implied.  See the License for the
 *  specific language governing permissions and limitations
 *  under the License.
 */
package jcompute.opencl;

import lombok.AccessLevel;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import jcompute.core.mem.JComputeArray;

@RequiredArgsConstructor(access = AccessLevel.PROTECTED)
public abstract class ClMem implements ClResource {

    public enum MemMode {
        MEM_READ_WRITE,
        MEM_READ_ONLY,
        MEM_WRITE_ONLY,
    }

    @Getter @Accessors(fluent = true) final ClContext context;
    @Getter @Accessors(fluent = true) final JComputeArray computeArray;

    /**
     * The number of elements contained in the underlying array.
     */
    public final long size() {
        return computeArray().shape().totalSize();
    }

    /**
     * The number bytes (required) for each element in the underlying array.
     */
    public final int sizeOf() {
        return computeArray().bytesPerElement();
    }

    protected abstract int releaseMemObject();

    @Override
    public final void free() {
        _Util.assertSuccess(releaseMemObject(), ()->
            String.format("failed to release memory object for context %s", context));
    }

}