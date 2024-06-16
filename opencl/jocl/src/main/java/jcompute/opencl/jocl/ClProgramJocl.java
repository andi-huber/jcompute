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

import java.util.LinkedList;
import java.util.List;

import org.jocl.CL;
import org.jocl.cl_device_id;
import org.jocl.cl_program;

import lombok.Getter;
import lombok.val;
import lombok.experimental.Accessors;

import jcompute.opencl.ClContext;
import jcompute.opencl.ClKernel;
import jcompute.opencl.ClProgram;
import jcompute.opencl.ClResource;

public class ClProgramJocl extends ClProgram {

    @Getter @Accessors(fluent = true) private final cl_program id;

    public ClProgramJocl(
            final cl_program id,
            final ClContext context,
            final List<ClResource> childResources) {
        super(context, childResources);
        this.id = id;
    }

    @Override
    public ClProgram build()  {
        return buildProgram(this);
    }

    @Override
    protected int releaseProgram() {
        return CL.clReleaseProgram(id());
    }

    @Override
    public ClKernel createKernel(final String kernelName) {
        return add(ClKernelJocl.createKernel(this, kernelName));
    }

    // -- HELPER

    /* Create Kernel program from the read in source */
    static ClProgram createProgram(final ClContextJocl context, final String source_str) {
        var ret_pointer = new int[1];
        val programId = CL.clCreateProgramWithSource(context.id(), 1,
                new String[]{source_str}, null, ret_pointer);
        int ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
            String.format("failed to create program for context %s",
                    context));
        return new ClProgramJocl(programId, context, new LinkedList<ClResource>());
    }

    /**
     * @implNote yet only supports contexts bound to only a single device
     */
    static ClProgram buildProgram(final ClProgramJocl program) {
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

    static String getBuildProgramInfo(final ClProgramJocl program, final int paramName) {
        val deviceId = ((ClDeviceJocl)program.getContext().getSingleDeviceElseFail()).id();
        return _Util.readString((a, b, c)->CL.clGetProgramBuildInfo(program.id(), deviceId, paramName, a, b, c));
    }

}
