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

import java.util.List;

import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.opencl._cl_context;
import org.bytedeco.opencl.global.OpenCL;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.core.mem.JComputeArray;
import jcompute.opencl.ClCommandQueue;
import jcompute.opencl.ClContext;
import jcompute.opencl.ClDevice;
import jcompute.opencl.ClMem;
import jcompute.opencl.ClMem.MemMode;
import jcompute.opencl.ClProgram;

public final class ClContextBd extends ClContext {

    @Getter @Accessors(fluent = true) private final _cl_context id;

    ClContextBd(final _cl_context id, final List<ClDevice> devices) {
        super(devices);
        this.id = id;
    }

    @Override
    protected ClCommandQueue createQueueInternal() {
        final ClContextBd context = this;
        var deviceId = ((ClDeviceBd)context.getSingleDeviceElseFail()).id();
        var properties = new LongPointer(new long[] {
//              OpenCL.CL_QUEUE_PROPERTIES,
//              | OpenCL.CL_QUEUE_ON_DEVICE
//              | OpenCL.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
              0}); // zero terminated list of queue creation properties
      // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateCommandQueueWithProperties.html
        var queueId = _Util.checkedApply(ret_pointer->
            OpenCL.clCreateCommandQueueWithProperties(
                context.id(),
                deviceId,
                properties,
                ret_pointer),
            ()->String.format("failed to create command-queue for context %s", context));
        return new ClCommandQueueBd(queueId, context);
    }

    @Override
    protected ClProgram createProgramInternal(final String programSource) {
        try(var sizeTPtr = new SizeTPointer(1); var src = new PointerPointer<>(programSource)){
            sizeTPtr.put(programSource.length());
            var programId = _Util.checkedApply(ret_pointer->
                OpenCL.clCreateProgramWithSource(this.id(), 1, src, sizeTPtr, ret_pointer),
                ()-> String.format("failed to create program for context %s", this));
            return new ClProgramBd(programId, this).build();
        }
    }

    @Override
    protected ClMem createMemoryInternal(final JComputeArray computeArray, final MemMode memMode) {
        var clMemMode = switch (memMode) {
            case MEM_READ_WRITE -> OpenCL.CL_MEM_READ_WRITE;
            case MEM_READ_ONLY -> OpenCL.CL_MEM_READ_ONLY;
            case MEM_WRITE_ONLY -> OpenCL.CL_MEM_WRITE_ONLY;
        };
        return createMemoryInternal(computeArray, clMemMode);
    }

    @Override
    protected int releaseContextIntenral() {
        return OpenCL.clReleaseContext(id());
    }

    // -- HELPER

    /**
     * Returns a new memory object for given context.
     */
    private ClMemBd createMemoryInternal(final JComputeArray computeArray, final long clMemMode) {
        long size = computeArray.shape().totalSize();
        int sizeOf = computeArray.bytesPerElement();

        var memId = _Util.checkedApply(ret_pointer->
            OpenCL.clCreateBuffer(this.id(), clMemMode, size * sizeOf, null, ret_pointer),
            ()->String.format("failed to create memory object (size=%d*%d) for context %s", sizeOf, size, this));
        return new ClMemBd(memId, this, computeArray);
    }


}
