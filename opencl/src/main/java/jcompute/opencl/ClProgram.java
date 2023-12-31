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

import java.util.LinkedList;
import java.util.List;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.opencl._cl_program;
import org.bytedeco.opencl.global.OpenCL;

import static org.bytedeco.opencl.global.OpenCL.clBuildProgram;
import static org.bytedeco.opencl.global.OpenCL.clCreateProgramWithSource;
import static org.bytedeco.opencl.global.OpenCL.clGetProgramBuildInfo;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.val;
import lombok.experimental.Accessors;

@RequiredArgsConstructor
public class ClProgram implements ClResource {

    @Getter @Accessors(fluent = true) private final _cl_program id;
    @Getter private final ClContext context;
    @Getter private final List<ClResource> childResources;

    public ClProgram build() {
        return buildProgram(this);
    }

    public ClKernel createKernel(final String kernelName) {
        return add(ClKernel.createKernel(this, kernelName));
    }

    @Override
    public void free() {

        childResources.forEach(ClResource::free);

        final int ret = OpenCL.clReleaseProgram(id());
        _Util.assertSuccess(ret, ()->
            String.format("failed to release program context %s", context));
    }

    @Override
    public String toString() {
        return String.format("ClProgram[%s]", context);
    }

    // -- HELPER

    /* Create Kernel program from the read in source */
    static ClProgram createProgram(final ClContext context, final String source_str) {
        try(val sizeTPtr = new SizeTPointer(1)){
            val ret_pointer = new IntPointer(1);
            val programId = clCreateProgramWithSource(context.id(), 1,
                    new PointerPointer<>(source_str), sizeTPtr.put(source_str.length()), ret_pointer);
            val ret = ret_pointer.get();
            _Util.assertSuccess(ret, ()->
                    String.format("failed to create program for context %s",
                            context));
            return new ClProgram(programId, context, new LinkedList<ClResource>());
        }
    }

    /**
     * @implNote yet only supports contexts bound to only a single device
     */
    static ClProgram buildProgram(final ClProgram program) {
        val deviceId = program.getContext().getSingleDeviceElseFail().id();

        /* Build Kernel Program */
        val ret = clBuildProgram(program.id(), 1, deviceId, null, null, null);
        _Util.assertSuccess(ret, ()->
        String.format("failed to build program %s%n"
                + "build-log: %s",
                program,
                getBuildProgramInfo(program, OpenCL.CL_PROGRAM_BUILD_LOG)
                ));

        return program;
    }


    //OpenCL.CL_PROGRAM_BUILD_LOG
    static String getBuildProgramInfo(final ClProgram program, final int paramName) {
        val deviceId = program.getContext().getSingleDeviceElseFail().id();
        val sizePointer = new SizeTPointer(1);
        clGetProgramBuildInfo(program.id(), deviceId, paramName, 0, null, sizePointer);
        final int size = (int)sizePointer.get();
        val buffer = new BytePointer(size);
        val ret = clGetProgramBuildInfo(program.id(), deviceId, paramName, size, buffer, null);
        _Util.assertSuccess(ret, ()->
            String.format("failed to getBuildProgramInfo %s", program));
        val result = new byte[size];
        buffer.get(result);
        return new String(result);
    }

    private <T extends ClResource> T add(final T resource) {
        this.childResources.add(0, resource);
        return resource;
    }


//    final PointerBuffer size = PointerBuffer.allocateDirect(1);
//
//    int ret = binding.clGetProgramBuildInfo(ID, device.ID, flag, 0, null, size);
//    if(ret != CL_SUCCESS) {
//        throw newException(ret, "on clGetProgramBuildInfo with "+device);
//    }
//
//    final ByteBuffer buffer = newDirectByteBuffer((int)size.get(0));
//
//    ret = binding.clGetProgramBuildInfo(ID, device.ID, flag, buffer.capacity(), buffer, null);
//    if(ret != CL_SUCCESS) {
//        throw newException(ret, "on clGetProgramBuildInfo with "+device);
//    }
//
//    return CLUtil.clString2JavaString(buffer, (int)size.get(0));



}
