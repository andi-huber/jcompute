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
package jcompute.opencl.bytedeco;

import org.bytedeco.opencl._cl_mem;
import org.bytedeco.opencl.global.OpenCL;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.core.mem.JComputeArray;
import jcompute.opencl.ClContext;
import jcompute.opencl.ClMem;

public final class ClMemBd extends ClMem {

    @Getter @Accessors(fluent = true) private final _cl_mem id;

    ClMemBd(
            final _cl_mem id,
            final ClContext context,
            final JComputeArray computeArray) {
        super(context, computeArray);
        this.id = id;
    }

    @Override
    protected int releaseMemObject() {
        return OpenCL.clReleaseMemObject(id());
    }

}
