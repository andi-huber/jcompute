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
import org.jocl.Pointer;

import lombok.experimental.UtilityClass;

import jcompute.core.mem.ByteArray;
import jcompute.core.mem.DoubleArray;
import jcompute.core.mem.JComputeArray;
import jcompute.core.mem.LongArray;
import jcompute.core.mem.ShortArray;
import jcompute.opencl.ClDevice.DeviceType;

@UtilityClass
class _Jocl {

    Pointer pointerOf(final JComputeArray jcomputeArray) {
        final Pointer pointer = switch (jcomputeArray) {
            case ByteArray array -> Pointer.to(array.toBuffer());
            case ShortArray array -> Pointer.to(array.toBuffer());
            case LongArray array -> Pointer.to(array.toBuffer());
            case DoubleArray array -> Pointer.to(array.toBuffer());
            default -> throw new IllegalArgumentException("Unexpected value: " + jcomputeArray.getClass());
        };
        return pointer;
    }

    static DeviceType fromClDeviceType(final int cl_device_type) {
        switch (cl_device_type) {
        case (int)CL.CL_DEVICE_TYPE_CPU: return DeviceType.CPU;
        case (int)CL.CL_DEVICE_TYPE_GPU: return DeviceType.GPU;
        case (int)CL.CL_DEVICE_TYPE_ACCELERATOR: return DeviceType.ACCELERATOR;
        default:
            return DeviceType.OTHER;
        }
    }

    // -- COMMAND QUEUE

    /* Transfer data to memory buffer */
    static ClCommandQueueJocl enqueueWriteBuffer(final ClCommandQueueJocl queue, final ClMemJocl memObj) {
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
    static ClCommandQueueJocl enqueueReadBuffer(final ClCommandQueueJocl queue, final ClMemJocl memObj) {
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
    static ClCommandQueueJocl enqueueNDRangeKernel(
            final ClCommandQueueJocl queue,
            final ClKernelJocl kernel,
            final int work_dim,
            final long[] global_work_size,
            final long[] local_work_size) {

        int ret = CL.clEnqueueNDRangeKernel(queue.id(), kernel.id(), work_dim, null,
                global_work_size, local_work_size, 0,
                null, null);
        _Util.assertSuccess(ret, ()->
            String.format("failed to enqueue Kernel for context %s", queue.getContext()));
        return queue;
    }

}
