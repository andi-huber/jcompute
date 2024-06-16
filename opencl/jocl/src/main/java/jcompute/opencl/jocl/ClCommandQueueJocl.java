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

import org.jocl.CL;
import org.jocl.cl_command_queue;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.core.shape.Shape;
import jcompute.opencl.ClCommandQueue;
import jcompute.opencl.ClContext;
import jcompute.opencl.ClKernel;
import jcompute.opencl.ClMem;

public final class ClCommandQueueJocl extends ClCommandQueue {

    @Getter @Accessors(fluent = true) private final cl_command_queue id;

    ClCommandQueueJocl(final cl_command_queue id, final ClContext context) {
        super(context);
        this.id = id;
    }

    @Override
    protected int flushQueue() {
        return CL.clFlush(id());
    }
    @Override
    protected int finishQueue() {
        return CL.clFinish(id());
    }
    @Override
    protected int releaseQueue() {
        return CL.clReleaseCommandQueue(id());
    }
    @Override
    protected int enqueueWriteBuffer(final ClMem memObj, final boolean blocking) {
        return CL.clEnqueueWriteBuffer(id(), ((ClMemJocl)memObj).id(),
                blocking,
                0,
                memObj.size() * memObj.sizeOf(),
                _Jocl.pointerOf(memObj.computeArray()),
                0,
                null, null);
    }
    @Override
    protected int enqueueReadBuffer(final ClMem memObj, final boolean blocking) {
        return CL.clEnqueueReadBuffer(id(), ((ClMemJocl)memObj).id(),
                blocking,
                0,
                memObj.size() * memObj.sizeOf(),
                _Jocl.pointerOf(memObj.computeArray()),
                0,
                null, null);
    }

    @Override
    public String toString() {
        return "addr: " + id.getNativePointer();
    }

    /**
     * localSize is auto
     */
    @Override
    public final ClCommandQueue enqueueNDRangeKernel(
            final ClKernel kernel,
            final Shape globalSize) {
        return _Jocl.enqueueNDRangeKernel(this, (ClKernelJocl)kernel, globalSize.dimensionCount(),
                new long[] {globalSize.sizeX(), globalSize.sizeY(), globalSize.sizeZ()},
                null);
    }

    @Override
    public final ClCommandQueue enqueueNDRangeKernel(
            final ClKernel kernel,
            final Shape globalSize,
            final Shape localSize) {
        return _Jocl.enqueueNDRangeKernel(this, (ClKernelJocl)kernel, globalSize.dimensionCount(),
                new long[] {globalSize.sizeX(), globalSize.sizeY(), globalSize.sizeZ()},
                new long[] {localSize.sizeX(), localSize.sizeY(), localSize.sizeZ()});
    }

}
