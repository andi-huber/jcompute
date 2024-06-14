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

import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Stack;

import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.opencl._cl_context;
import org.bytedeco.opencl.global.OpenCL;

import static org.bytedeco.opencl.global.OpenCL.clCreateContext;
import static org.bytedeco.opencl.global.OpenCL.clReleaseContext;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.val;
import lombok.experimental.Accessors;

import jcompute.core.mem.JComputeArray;

@RequiredArgsConstructor
public class ClContext implements AutoCloseable {

    @Getter @Accessors(fluent = true) private final _cl_context id;
    @Getter private final List<ClDevice> devices;
    @Getter private final Stack<ClResource> childResources;

    public ClCommandQueue createQueue() {
        return add(ClCommandQueue.createQueue(this));
    }

    public ClProgram createProgram(final String programSource) {
        return add(ClProgram.createProgram(this, programSource).build());
    }

    public ClMem createMemoryReadWrite(final JComputeArray array) {
        return add(ClMem.createMemory(this, array, OpenCL.CL_MEM_READ_WRITE));
    }
    public ClMem createMemoryReadOnly(final JComputeArray array) {
        return add(ClMem.createMemory(this, array, OpenCL.CL_MEM_READ_ONLY));
    }
    public ClMem createMemoryWriteOnly(final JComputeArray array) {
        return add(ClMem.createMemory(this, array, OpenCL.CL_MEM_WRITE_ONLY));
    }

    public ClMem createMemoryReadWrite(final Pointer mem) {
        return add(ClMem.createMemory(this, mem, OpenCL.CL_MEM_READ_WRITE));
    }
    public ClMem createMemoryReadOnly(final Pointer mem) {
        return add(ClMem.createMemory(this, mem, OpenCL.CL_MEM_READ_ONLY));
    }
    public ClMem createMemoryWriteOnly(final Pointer mem) {
        return add(ClMem.createMemory(this, mem, OpenCL.CL_MEM_WRITE_ONLY));
    }

    public ClDevice getSingleDeviceElseFail() {
        if(getDevices().size()!=1) {
            throw new IllegalArgumentException(String.format(
                    "context %s is required to be bound to exaclty one device", this));
        }
        return getDevices().get(0);
    }

    @Override
    public void close() {
        while(!childResources.isEmpty()) {
            childResources.pop().free();
        }
        final int ret = clReleaseContext(id());
        _Util.assertSuccess(ret, ()->
            String.format("failed to release context for devices %s", devices));
    }

    @Override
    public String toString() {
        return String.format("ClContext[%s]", devices);
    }

    // -- UTILITY

    @SneakyThrows
    public ClProgram createProgram(final Class<?> cls, final String resourceName) {
        try(val is = cls.getResourceAsStream(resourceName)){
            val source = _Util.read(is, StandardCharsets.UTF_8);
            return createProgram(source);
        }
    }

    // -- HELPER

    private <T extends ClResource> T add(final T resource) {
        childResources.push(resource);
        return resource;
    }

    /**
     * Returns a context bound to a single device.
     * <p>
     * @apiNote OpenCL supports binding to multiple devices as well
     */
    static ClContext createContext(final ClDevice device) {
        val ret_pointer = new IntPointer(1);
        val contextId = clCreateContext(null, 1, device.id(), null, null, ret_pointer);
        val ret = ret_pointer.get();
        _Util.assertSuccess(ret, ()->
                String.format("failed to create context for device %s", device.getName()));
        return new ClContext(contextId, List.of(device), new Stack<ClResource>());
    }


}
