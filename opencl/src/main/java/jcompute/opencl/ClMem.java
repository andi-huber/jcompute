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

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.opencl._cl_mem;
import org.bytedeco.opencl.global.OpenCL;

import static org.bytedeco.opencl.global.OpenCL.clCreateBuffer;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.val;
import lombok.experimental.Accessors;

@RequiredArgsConstructor
public class ClMem implements ClResource {

    @Getter @Accessors(fluent = true) private final _cl_mem id;
    @Getter private final ClContext context;
    @Getter @Accessors(fluent = true) private final long size;
    @Getter @Accessors(fluent = true) private final int sizeOf;
    @Getter private final Pointer pointer;

    @Override
    public void free() {
        final int ret = OpenCL.clReleaseMemObject(id());
        _Util.assertSuccess(ret, ()->
            String.format("failed to release memory object for context %s", context));
    }

    // -- HELPER

    /**
     * Returns a new memory object for given context.
     * @param options
     * @param size
     */
    static ClMem createMemory(final ClContext context, final Pointer pointer, final int options) {
        final long size = pointer.capacity();
        final int sizeOf = Loader.sizeof(pointer.getClass());

        val ret_pointer = new IntPointer(1);
        val memId = clCreateBuffer(context.id(), options,
                size * sizeOf, null, ret_pointer);
        val ret = ret_pointer.get();
        _Util.assertSuccess(ret, ()->
                String.format("failed to create memory object for context %s", context));
        return new ClMem(memId, context, size, sizeOf, pointer);
    }

}
