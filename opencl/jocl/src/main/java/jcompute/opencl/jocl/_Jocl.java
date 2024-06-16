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

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;
import org.jocl.cl_queue_properties;

import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetPlatformIDs;

import lombok.Getter;
import lombok.val;
import lombok.experimental.UtilityClass;

import jcompute.core.mem.ByteArray;
import jcompute.core.mem.DoubleArray;
import jcompute.core.mem.JComputeArray;
import jcompute.core.mem.LongArray;
import jcompute.core.mem.ShortArray;
import jcompute.opencl.ClBinding;
import jcompute.opencl.ClCommandQueue;
import jcompute.opencl.ClDevice;
import jcompute.opencl.ClDevice.DeviceType;
import jcompute.opencl.ClPlatform;
import jcompute.opencl.ClProgram;
import jcompute.opencl.ClResource;

@UtilityClass
class _Jocl {

    // -- BINDING

    @Getter(lazy = true)
    private final static ClBinding binding = createBinding();

    private ClBinding createBinding() {
        final var platforms = _Jocl.createPlatforms();
        return () -> platforms;
    }

    // -- PLATFORMS

    /**
     * Lists all available OpenCL implementations.
     */
    static List<ClPlatform> createPlatforms() {

        val outPlatformCount = new int[1];
        // count available OpenCL platforms
        _Util.assertSuccess(
                clGetPlatformIDs(0, null, outPlatformCount),
                ()->"failed to call clGetPlatformIDs");

        final int platformCount = outPlatformCount[0];
        val platformBuffer = new cl_platform_id[platformCount];

        // fetch available OpenCL platforms
        _Util.assertSuccess(
                clGetPlatformIDs(platformCount, platformBuffer, null),
                ()->"failed to call clGetPlatformIDs");

        val platforms = new ArrayList<ClPlatformJocl>(platformCount);
        for (int i = 0; i < platformCount; i++) {
            platforms.add(
                    new ClPlatformJocl(i, platformBuffer[i]));
        }

        return Collections.unmodifiableList(platforms);
    }

    // -- DEVICE

    DeviceType fromClDeviceType(final int cl_device_type) {
        switch (cl_device_type) {
        case (int)CL.CL_DEVICE_TYPE_CPU: return DeviceType.CPU;
        case (int)CL.CL_DEVICE_TYPE_GPU: return DeviceType.GPU;
        case (int)CL.CL_DEVICE_TYPE_ACCELERATOR: return DeviceType.ACCELERATOR;
        default:
            return DeviceType.OTHER;
        }
    }

    List<ClDevice> listDevices(final ClPlatformJocl platform) {
        val platformId = platform.id();
        // Obtain the number of devices for the platform
        final int[] numDevicesRef = new int[1];
        clGetDeviceIDs(platformId, CL.CL_DEVICE_TYPE_ALL, 0, null, numDevicesRef);
        final int deviceCount = numDevicesRef[0];

        val deviceIds = new cl_device_id[deviceCount];
        clGetDeviceIDs(platformId, CL.CL_DEVICE_TYPE_ALL, deviceCount, deviceIds, (int[])null);

        val devices = new ArrayList<ClDevice>(deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            devices.add(
                    new ClDeviceJocl(deviceIds[i], null, i));
        }

        return Collections.unmodifiableList(devices);
    }

    // -- CONTEXT

    /**
     * Returns a context bound to a single device.
     * <p>
     * @apiNote OpenCL supports binding to multiple devices as well
     */
    static ClContextJocl createContext(final ClDeviceJocl device) {
        val ret_pointer = new int[1];
        val contextId = CL.clCreateContext(null, 1, new cl_device_id[]{device.id()}, null, null, ret_pointer);
        val ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
                String.format("failed to create context for device %s", device.getName()));
        return new ClContextJocl(contextId, List.of(device));
    }

    // -- PROGRAM

    /**
     * @implNote yet only supports contexts bound to only a single device
     */
    ClProgram buildProgram(final ClProgramJocl program) {
        val deviceId = ((ClDeviceJocl)program.getContext().getSingleDeviceElseFail()).id();

        // https://www.intel.com/content/www/us/en/docs/opencl-sdk/developer-reference-processor-graphics/2015-1/optimization-options.html
        String options = null; // all enabled by default
                //"-cl-opt-disable";
                //"-cl-mad-enable";

        /* Build Kernel Program */
        int ret = CL.clBuildProgram(program.id(), 1, new cl_device_id[] {deviceId}, options, null, null);
        _Util.assertSuccess(ret, ()->
            String.format("failed to build program %s%n"
                    + "build-log: %s",
                    program,
                    getBuildProgramInfo(program, CL.CL_PROGRAM_BUILD_LOG)
                    ));

        return program;
    }

    String getBuildProgramInfo(final ClProgramJocl program, final int paramName) {
        val deviceId = ((ClDeviceJocl)program.getContext().getSingleDeviceElseFail()).id();
        return _Util.readString((a, b, c)->CL.clGetProgramBuildInfo(program.id(), deviceId, paramName, a, b, c));
    }

    /* Create Kernel program from the read in source */
    ClProgramJocl createProgram(final ClContextJocl context, final String source_str) {
        var ret_pointer = new int[1];
        val programId = CL.clCreateProgramWithSource(context.id(), 1,
                new String[]{source_str}, null, ret_pointer);
        int ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
            String.format("failed to create program for context %s",
                    context));
        return new ClProgramJocl(programId, context, new LinkedList<ClResource>());
    }

    // -- KERNEL

    /**
     * Returns a new OpenCL kernel for given program.
     */
    ClKernelJocl createKernel(final ClProgramJocl program, final String kernelName) {
        val ret_pointer = new int[1];
        val kernelId = CL.clCreateKernel(program.id(), kernelName, ret_pointer);
        int ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
                String.format("failed to create kernel '%s' for program %s", kernelName, program));
        return new ClKernelJocl(kernelId, program, kernelName);
    }

    // -- MEMORY

    /**
     * Returns a new memory object for given context.
     */
    ClMemJocl createMemory(final ClContextJocl context, final JComputeArray computeArray, final long options) {
        long size = computeArray.shape().totalSize();
        int sizeOf = computeArray.bytesPerElement();

        val ret_pointer = new int[1];
        val memId = CL.clCreateBuffer(context.id(), options,
                size * sizeOf, null, ret_pointer);
        val ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
                String.format("failed to create memory object (size=%d*%d) for context %s", sizeOf, size, context));

        return new ClMemJocl(memId, context, computeArray);
    }

    // -- COMMAND QUEUE

    /**
     * Returns a new command queue for given context.
     * @implNote yet only supports contexts bound to only a single device
     */
    ClCommandQueue createQueue(final ClContextJocl context) {
        val deviceId = ((ClDeviceJocl)context.getSingleDeviceElseFail()).id();
        // zero terminated list of queue creation properties
        // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateCommandQueueWithProperties.html
        cl_queue_properties properties = new cl_queue_properties();
        int[] ret_pointer = new int[1];
        val queueId = CL.clCreateCommandQueueWithProperties(context.id(), deviceId, properties, ret_pointer );
        val ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
                String.format("failed to create command-queue for context %s", context));
        return new ClCommandQueueJocl(queueId, context);
    }

    // -- POINTER

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

}
