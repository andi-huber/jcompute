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
import org.jocl.cl_device_id;
import org.jocl.cl_program;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.opencl.ClContext;
import jcompute.opencl.ClKernel;
import jcompute.opencl.ClProgram;

public final class ClProgramJocl extends ClProgram {

    @Getter @Accessors(fluent = true) private final cl_program id;

    ClProgramJocl(
            final cl_program id,
            final ClContext context) {
        super(context);
        this.id = id;
    }

    /**
     * @implNote yet only supports contexts bound to only a single device
     */
    @Override
    public ClProgram build()  {
        var deviceId = ((ClDeviceJocl)getContext().getSingleDeviceElseFail()).id();

        // https://www.intel.com/content/www/us/en/docs/opencl-sdk/developer-reference-processor-graphics/2015-1/optimization-options.html
        String options = null; // all enabled by default
                //"-cl-opt-disable";
                //"-cl-mad-enable";

        /* Build Kernel Program */
        _Util.assertSuccess(
                CL.clBuildProgram(id(), 1, new cl_device_id[] {deviceId}, options, null, null),
                ()->String.format("failed to build program %s%n"
                    + "build-log: %s",
                    this,
                    getBuildProgramInfo(CL.CL_PROGRAM_BUILD_LOG)
                    ));
        return this;
    }

    @Override
    protected int releaseProgramInternal() {
        return CL.clReleaseProgram(id());
    }

    @Override
    protected ClKernel createKernelInternal(final String kernelName) {
        var kernelId = _Util.checkedApply(ret_pointer->
                CL.clCreateKernel(this.id(), kernelName, ret_pointer),
                ()->String.format("failed to create kernel '%s' for program %s", kernelName, this));
        return new ClKernelJocl(kernelId, this, kernelName);
    }

    // -- HELPER

    private String getBuildProgramInfo(final int paramName) {
        var deviceId = ((ClDeviceJocl)getContext().getSingleDeviceElseFail()).id();
        return _Util.readString((a, b, c)->CL.clGetProgramBuildInfo(id(), deviceId, paramName, a, b, c));
    }

}
