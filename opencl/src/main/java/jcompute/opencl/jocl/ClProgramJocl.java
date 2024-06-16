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
import org.jocl.cl_program;

import lombok.val;

import jcompute.opencl.ClContext;
import jcompute.opencl.ClProgram;
import jcompute.opencl.ClResource;

public class ClProgramJocl extends ClProgram {

    //@Getter @Accessors(fluent = true) private final cl_program id;
    //@Getter private final ClContext context;
    //@Getter private final List<ClResource> childResources;

    public ClProgramJocl(
            final cl_program id,
            final ClContext context,
            final List<ClResource> childResources) {
        super(id, context, childResources);
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
        return new ClProgramJocl(programId, context, new LinkedList<ClResource>());
    }

}
