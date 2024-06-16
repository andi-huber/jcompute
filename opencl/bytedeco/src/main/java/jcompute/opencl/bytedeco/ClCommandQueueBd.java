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
package jcompute.opencl.bytedeco;

import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.opencl._cl_command_queue;
import org.bytedeco.opencl.global.OpenCL;

import static org.bytedeco.opencl.global.OpenCL.CL_FALSE;
import static org.bytedeco.opencl.global.OpenCL.CL_TRUE;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.core.mem.ByteArray;
import jcompute.core.mem.DoubleArray;
import jcompute.core.mem.JComputeArray;
import jcompute.core.mem.LongArray;
import jcompute.core.mem.ShortArray;
import jcompute.core.shape.Shape;
import jcompute.opencl.ClCommandQueue;
import jcompute.opencl.ClContext;
import jcompute.opencl.ClKernel;
import jcompute.opencl.ClMem;
import jcompute.opencl.bytedeco.util.PointerUtils;

public final class ClCommandQueueBd extends ClCommandQueue {

    @Getter @Accessors(fluent = true) private final _cl_command_queue id;

    ClCommandQueueBd(final _cl_command_queue id, final ClContext context) {
        super(context);
        this.id = id;
    }

    @Override
    protected int flushQueue() {
        return OpenCL.clFlush(id());
    }
    @Override
    protected int finishQueue() {
        return OpenCL.clFinish(id());
    }
    @Override
    protected int releaseQueue() {
        return OpenCL.clReleaseCommandQueue(id());
    }
    @Override
    protected int enqueueWriteBuffer(final ClMem memObj, final boolean blocking) {
        return OpenCL.clEnqueueWriteBuffer(id(), ((ClMemBd)memObj).id(),
                blocking ? CL_TRUE : CL_FALSE,
                0,
                memObj.size() * memObj.sizeOf(),
                pointerOf(memObj.computeArray()),
                0,
                (PointerPointer<?>)null, null);
    }
    @Override
    protected int enqueueReadBuffer(final ClMem memObj, final boolean blocking) {
        return OpenCL.clEnqueueReadBuffer(id(), ((ClMemBd)memObj).id(),
                blocking ? CL_TRUE : CL_FALSE,
                0,
                memObj.size() * memObj.sizeOf(),
                pointerOf(memObj.computeArray()),
                0,
                (PointerPointer<?>)null, null);
    }

    @Override
    public String toString() {
        return "addr: " + id.address();
    }

    @Override
    protected int enqueueNDRangeKernel(
            final ClKernel kernel, final int work_dim,
            final Shape globalSize, final Shape localSize) {

        var global_work_size = new SizeTPointer(globalSize.sizeX(), globalSize.sizeY(), globalSize.sizeZ());
        var local_work_size = localSize!=null
                ? new SizeTPointer(localSize.sizeX(), localSize.sizeY(), localSize.sizeZ())
                : null;

        return OpenCL.clEnqueueNDRangeKernel(id(), ((ClKernelBd)kernel).id(), work_dim, null,
                global_work_size, local_work_size, 0,
                (PointerPointer<?>)null, null);
    }

    // -- HELPER

    private static Pointer pointerOf(final JComputeArray jcomputeArray) {
        final Pointer pointer = switch (jcomputeArray) {
            case ByteArray array -> PointerUtils.pointer(array);
            case ShortArray array -> PointerUtils.pointer(array);
            case LongArray array -> PointerUtils.pointer(array);
            case DoubleArray array -> PointerUtils.pointer(array);
            default -> throw new IllegalArgumentException("Unexpected value: " + jcomputeArray.getClass());
        };
        return pointer;
    }

}
