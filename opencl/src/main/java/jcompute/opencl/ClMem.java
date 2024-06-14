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

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_mem;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.val;
import lombok.experimental.Accessors;

import jcompute.core.mem.ByteArray;
import jcompute.core.mem.DoubleArray;
import jcompute.core.mem.JComputeArray;
import jcompute.core.mem.LongArray;
import jcompute.core.mem.ShortArray;

@RequiredArgsConstructor
public class ClMem implements ClResource {

    @Getter @Accessors(fluent = true) private final cl_mem id;
    @Getter private final ClContext context;
    /**
     * The number of elements contained in the underlying array.
     */
    @Getter @Accessors(fluent = true) private final long size;
    /**
     * The number bytes (required) for each element in the underlying array.
     */
    @Getter @Accessors(fluent = true) private final int sizeOf;
    @Getter private final Pointer pointer;

    @Override
    public void free() {
        final int ret = CL.clReleaseMemObject(id());
        _Util.assertSuccess(ret, ()->
            String.format("failed to release memory object for context %s", context));
    }

    // -- HELPER

    /**
     * Returns a new memory object for given context.
     */
    static ClMem createMemory(final ClContext context, final JComputeArray jcomputeArray, final long options) {
        int sizeOf = jcomputeArray.bytesPerElement();
        long size = jcomputeArray.shape().totalSize();

        val ret_pointer = new int[1];
        val memId = CL.clCreateBuffer(context.id(), options,
                size * sizeOf, null, ret_pointer);
        val ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
                String.format("failed to create memory object (size=%d*%d) for context %s", sizeOf, size, context));

        final Pointer pointer = switch (jcomputeArray) {
            case ByteArray array -> Pointer.to(array.toBuffer());
            case ShortArray array -> Pointer.to(array.toBuffer());
            case LongArray array -> Pointer.to(array.toBuffer());
            case DoubleArray array -> Pointer.to(array.toBuffer());
            default -> throw new IllegalArgumentException("Unexpected value: " + jcomputeArray.getClass());
        };

        return new ClMem(memId, context, size, sizeOf, pointer);
    }

}

