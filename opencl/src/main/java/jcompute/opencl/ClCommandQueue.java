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
import org.jocl.cl_command_queue;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.experimental.Accessors;

import jcompute.core.shape.Shape;

@RequiredArgsConstructor
public abstract class ClCommandQueue implements ClResource {

    @Getter @Accessors(fluent = true) private final cl_command_queue id;
    @Getter private final ClContext context;

    public final ClCommandQueue enqueueWriteBuffer(final ClMem memObj) {
        return enqueueWriteBuffer(this, memObj);
    }

    public final ClCommandQueue enqueueReadBuffer(final ClMem memObj) {
        return enqueueReadBuffer(this, memObj);
    }

    public final ClCommandQueue enqueueNDRangeKernel(
            final ClKernel kernel,
            final Shape globalSize,
            final Shape localSize) {
        return enqueueNDRangeKernel(this, kernel, globalSize.dimensionCount(),
                new long[] {globalSize.sizeX(), globalSize.sizeY(), globalSize.sizeZ()},
                new long[] {localSize.sizeX(), localSize.sizeY(), localSize.sizeZ()});
    }

    public final ClCommandQueue flush() {
        final int ret = CL.clFlush(id());
        _Util.assertSuccess(
                ret, ()->
                    String.format("failed to flush command queue for context %s", context));
        return this;
    }

    public final ClCommandQueue finish() {
        final int ret = CL.clFinish(id());
        _Util.assertSuccess(
                ret, ()->
                    String.format("failed to finish command queue for context %s", context));
        return this;
    }

    /**
     * localSize is auto
     */
    public final ClCommandQueue enqueueNDRangeKernel(
            final ClKernel kernel,
            final Shape globalSize) {
        return enqueueNDRangeKernel(this, kernel, globalSize.dimensionCount(),
                new long[] {globalSize.sizeX(), globalSize.sizeY(), globalSize.sizeZ()},
                null);
    }

    @Override
    public final void free() {
        flush();
        finish();

        int ret = CL.clReleaseCommandQueue(id());
        _Util.assertSuccess(
                ret, ()->
                    String.format("failed to release command queue for context %s", context));
    }

    // -- HELPER

    /* Transfer data to memory buffer */
    static ClCommandQueue enqueueWriteBuffer(final ClCommandQueue queue, final ClMem memObj) {
        int ret = CL.clEnqueueWriteBuffer(queue.id(), memObj.id(),
                true, // blocking write
                0,
                memObj.size() * memObj.sizeOf(),
                _Jocl.pointerOf(memObj.computeArray()),
                0,
                null, null);
        _Util.assertSuccess(ret, ()->
            String.format("failed to enqueue WriteBuffer for context %s", queue.getContext()));
        return queue;
    }

    /* Transfer result from the memory buffer */
    static ClCommandQueue enqueueReadBuffer(final ClCommandQueue queue, final ClMem memObj) {
        int ret = CL.clEnqueueReadBuffer(queue.id(), memObj.id(),
                true, // blocking read
                0,
                memObj.size() * memObj.sizeOf(),
                _Jocl.pointerOf(memObj.computeArray()),
                0,
                null, null);
        _Util.assertSuccess(ret, ()->
            String.format("failed to enqueue ReadBuffer for context %s", queue.getContext()));
        return queue;
    }

    /**
     * Execute OpenCL kernel
     * @param queue
     * @param kernel
     * @param work_dim - number of dimensions used to specify the global work-items and work-items in
            the work-group
     * @param global_work_size
     * @param local_work_size
     */
    static ClCommandQueue enqueueNDRangeKernel(
            final ClCommandQueue queue,
            final ClKernel kernel,
            final int work_dim,
            final long[] global_work_size,
            final long[] local_work_size) {

        int ret = CL.clEnqueueNDRangeKernel(queue.id, kernel.id(), work_dim, null,
                global_work_size, local_work_size, 0,
                null, null);
        _Util.assertSuccess(ret, ()->
            String.format("failed to enqueue Kernel for context %s", queue.getContext()));
        return queue;
    }

}
