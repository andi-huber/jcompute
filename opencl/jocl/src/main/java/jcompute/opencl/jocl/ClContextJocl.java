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
package jcompute.opencl.jocl;

import java.util.List;

import org.jocl.CL;
import org.jocl.cl_context;
import org.jocl.cl_queue_properties;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.core.mem.JComputeArray;
import jcompute.opencl.ClCommandQueue;
import jcompute.opencl.ClContext;
import jcompute.opencl.ClDevice;
import jcompute.opencl.ClMem;
import jcompute.opencl.ClMem.MemMode;
import jcompute.opencl.ClProgram;

public final class ClContextJocl extends ClContext {

    @Getter @Accessors(fluent = true) private final cl_context id;

    ClContextJocl(final cl_context id, final List<ClDevice> devices) {
        super(devices);
        this.id = id;
    }

    @Override
    protected ClCommandQueue createQueueInternal() {
        final ClContextJocl context = this;
        var deviceId = ((ClDeviceJocl)context.getSingleDeviceElseFail()).id();
        var queueId = _Util.checkedApply(ret_pointer->
            CL.clCreateCommandQueueWithProperties(
                context.id(),
                deviceId,
                // zero terminated list of queue creation properties
                // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateCommandQueueWithProperties.html
                new cl_queue_properties(),
                ret_pointer),
            ()->String.format("failed to create command-queue for context %s", context));
        return new ClCommandQueueJocl(queueId, context);
    }

    @Override
    protected ClProgram createProgramInternal(final String programSource) {
        var programId = _Util.checkedApply(ret_pointer->
            CL.clCreateProgramWithSource(this.id(), 1, new String[]{programSource}, null, ret_pointer),
            ()-> String.format("failed to create program for context %s", this));
        return new ClProgramJocl(programId, this).build();
    }

    @Override
    protected ClMem createMemoryInternal(final JComputeArray computeArray, final MemMode memMode) {
        var clMemMode = switch (memMode) {
            case MEM_READ_WRITE -> CL.CL_MEM_READ_WRITE;
            case MEM_READ_ONLY -> CL.CL_MEM_READ_ONLY;
            case MEM_WRITE_ONLY -> CL.CL_MEM_WRITE_ONLY;
        };
        return createMemoryInternal(computeArray, clMemMode);
    }

    @Override
    protected int releaseContextIntenral() {
        return CL.clReleaseContext(id());
    }

    // -- HELPER

    /**
     * Returns a new memory object for given context.
     */
    private ClMemJocl createMemoryInternal(final JComputeArray computeArray, final long clMemMode) {
        long size = computeArray.shape().totalSize();
        int sizeOf = computeArray.bytesPerElement();

        var memId = _Util.checkedApply(ret_pointer->
            CL.clCreateBuffer(this.id(), clMemMode, size * sizeOf, null, ret_pointer),
            ()->String.format("failed to create memory object (size=%d*%d) for context %s", sizeOf, size, this));
        return new ClMemJocl(memId, this, computeArray);
    }

}
