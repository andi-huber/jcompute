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

import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.opencl._cl_command_queue;
import org.bytedeco.opencl.global.OpenCL;

import static org.bytedeco.opencl.global.OpenCL.CL_TRUE;
import static org.bytedeco.opencl.global.OpenCL.clEnqueueNDRangeKernel;
import static org.bytedeco.opencl.global.OpenCL.clEnqueueReadBuffer;
import static org.bytedeco.opencl.global.OpenCL.clEnqueueWriteBuffer;
import static org.bytedeco.opencl.global.OpenCL.clFinish;
import static org.bytedeco.opencl.global.OpenCL.clFlush;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.val;
import lombok.experimental.Accessors;

import jcompute.core.shape.Shape;

@RequiredArgsConstructor
public class ClCommandQueue implements ClResource {

    @Getter @Accessors(fluent = true) private final _cl_command_queue id;
    @Getter private final ClContext context;

    public ClCommandQueue enqueueWriteBuffer(final ClMem memObj) {
        return enqueueWriteBuffer(this, memObj);
    }

    public ClCommandQueue enqueueReadBuffer(final ClMem memObj) {
        return enqueueReadBuffer(this, memObj);
    }

    public ClCommandQueue enqueueNDRangeKernel(
            final ClKernel kernel,
            final Shape globalSize,
            final Shape localSize) {
        return enqueueNDRangeKernel(this, kernel, globalSize.dimensionCount(),
                new SizeTPointer(globalSize.sizeX(), globalSize.sizeY(), globalSize.sizeZ()),
                new SizeTPointer(localSize.sizeX(), localSize.sizeY(), localSize.sizeZ()));
    }

    public ClCommandQueue flush() {
        final int ret = clFlush(id());
        _Util.assertSuccess(
                ret, ()->
                    String.format("failed to flush command queue for context %s", context));
        return this;
    }

    public ClCommandQueue finish() {
        final int ret = clFinish(id());
        _Util.assertSuccess(
                ret, ()->
                    String.format("failed to finish command queue for context %s", context));
        return this;
    }

    /**
     * localSize is auto
     */
    public ClCommandQueue enqueueNDRangeKernel(
            final ClKernel kernel,
            final Shape globalSize) {
        return enqueueNDRangeKernel(this, kernel, globalSize.dimensionCount(),
                new SizeTPointer(globalSize.sizeX(), globalSize.sizeY(), globalSize.sizeZ()),
                null);
    }

    @Override
    public void free() {
        flush();
        finish();

        int ret = OpenCL.clReleaseCommandQueue(id());
        _Util.assertSuccess(
                ret, ()->
                    String.format("failed to release command queue for context %s", context));
    }

    // -- HELPER

    /**
     * Returns a new command queue for given context.
     * @implNote yet only supports contexts bound to only a single device
     */
    static ClCommandQueue createQueue(final ClContext context) {
        val deviceId = context.getSingleDeviceElseFail().id();
        val properties = new long[] {
//                OpenCL.CL_QUEUE_PROPERTIES,
//                | OpenCL.CL_QUEUE_ON_DEVICE
//                | OpenCL.CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                0}; // zero terminated list of queue creation properties
        // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateCommandQueueWithProperties.html
        val ret_pointer = new int[1];
        val queueId = OpenCL.clCreateCommandQueueWithProperties(context.id(), deviceId, properties, ret_pointer);
        val ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
                String.format("failed to create command-queue for context %s", context));
        return new ClCommandQueue(queueId, context);
    }

    /* Transfer data to memory buffer */
    static ClCommandQueue enqueueWriteBuffer(final ClCommandQueue queue, final ClMem memObj) {
        val ret = clEnqueueWriteBuffer(queue.id(), memObj.id(),
                CL_TRUE,
                0,
                memObj.size() * memObj.sizeOf(),
                memObj.getPointer(),
                0,
                (PointerPointer<?>)null, null);
        _Util.assertSuccess(ret, ()->
            String.format("failed to enqueue WriteBuffer for context %s", queue.getContext()));
        return queue;
    }

    /* Transfer result from the memory buffer */
    static ClCommandQueue enqueueReadBuffer(final ClCommandQueue queue, final ClMem memObj) {
        val ret = clEnqueueReadBuffer(queue.id(), memObj.id(),
                CL_TRUE,
                0,
                memObj.size() * memObj.sizeOf(),
                memObj.getPointer(),
                0,
                (PointerPointer<?>)null, null);
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
            final SizeTPointer global_work_size,
            final SizeTPointer local_work_size) {
        val ret = clEnqueueNDRangeKernel(queue.id, kernel.id(), work_dim, null,
                global_work_size, local_work_size, 0,
                (PointerPointer<?>)null, null);
        _Util.assertSuccess(ret, ()->
            String.format("failed to enqueue Kernel for context %s", queue.getContext()));
        return queue;
    }

}
