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
import org.jocl.cl_context;
import org.jocl.cl_device_id;

import lombok.Getter;
import lombok.val;
import lombok.experimental.Accessors;

import jcompute.core.mem.JComputeArray;
import jcompute.opencl.ClCommandQueue;
import jcompute.opencl.ClContext;
import jcompute.opencl.ClDevice;
import jcompute.opencl.ClMem;
import jcompute.opencl.ClProgram;

public class ClContextJocl extends ClContext {

    @Getter @Accessors(fluent = true) private final cl_context id;

    public ClContextJocl(final cl_context id, final List<ClDevice> devices) {
        super(devices);
        this.id = id;
    }

    @Override
    public ClCommandQueue createQueue() {
        return add(ClCommandQueueJocl.createQueue(this));
    }

    @Override
    public ClProgram createProgram(final String programSource) {
        return add(ClProgramJocl.createProgram(this, programSource).build());
    }

    @Override
    public ClMem createMemoryReadWrite(final JComputeArray array) {
        return add(ClMemJocl.createMemory(this, array, CL.CL_MEM_READ_WRITE));
    }
    @Override
    public ClMem createMemoryReadOnly(final JComputeArray array) {
        return add(ClMemJocl.createMemory(this, array, CL.CL_MEM_READ_ONLY));
    }
    @Override
    public ClMem createMemoryWriteOnly(final JComputeArray array) {
        return add(ClMemJocl.createMemory(this, array, CL.CL_MEM_WRITE_ONLY));
    }

    // -- UTILITY

    /**
     * Returns a context bound to a single device.
     * <p>
     * @apiNote OpenCL supports binding to multiple devices as well
     */
    static ClContextJocl createContext(final ClDevice device) {
        val ret_pointer = new int[1];
        val contextId = CL.clCreateContext(null, 1, new cl_device_id[]{device.id()}, null, null, ret_pointer);
        val ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
                String.format("failed to create context for device %s", device.getName()));
        return new ClContextJocl(contextId, List.of(device));
    }

}
