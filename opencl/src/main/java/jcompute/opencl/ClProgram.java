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

import org.jocl.CL;
import org.jocl.cl_device_id;
import org.jocl.cl_program;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.val;
import lombok.experimental.Accessors;

@RequiredArgsConstructor
public class ClProgram implements ClResource {

    @Getter @Accessors(fluent = true) private final cl_program id;
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
        final int ret = CL.clReleaseProgram(id());
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
        var ret_pointer = new int[1];
        val programId = CL.clCreateProgramWithSource(context.id(), 1,
                new String[]{source_str}, null, ret_pointer);
        int ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
            String.format("failed to create program for context %s",
                    context));
        return new ClProgram(programId, context, new LinkedList<ClResource>());
    }

    /**
     * @implNote yet only supports contexts bound to only a single device
     */
    static ClProgram buildProgram(final ClProgram program) {
        val deviceId = program.getContext().getSingleDeviceElseFail().id();

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


    //OpenCL.CL_PROGRAM_BUILD_LOG
    static String getBuildProgramInfo(final ClProgram program, final int paramName) {
        val deviceId = program.getContext().getSingleDeviceElseFail().id();
        return _Util.readString((a, b, c)->CL.clGetProgramBuildInfo(program.id(), deviceId, paramName, a, b, c));
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
