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
import org.jocl.cl_program;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.opencl.ClContext;
import jcompute.opencl.ClKernel;
import jcompute.opencl.ClProgram;
import jcompute.opencl.ClResource;

public final class ClProgramJocl extends ClProgram {

    @Getter @Accessors(fluent = true) private final cl_program id;

    ClProgramJocl(
            final cl_program id,
            final ClContext context,
            final List<ClResource> childResources) {
        super(context, childResources);
        this.id = id;
    }

    @Override
    public ClProgram build()  {
        return _Jocl.buildProgram(this);
    }

    @Override
    protected int releaseProgram() {
        return CL.clReleaseProgram(id());
    }

    @Override
    public ClKernel createKernel(final String kernelName) {
        return add(_Jocl.createKernel(this, kernelName));
    }

}
