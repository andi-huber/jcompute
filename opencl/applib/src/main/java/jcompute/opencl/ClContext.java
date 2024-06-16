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

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import lombok.val;
import lombok.experimental.Accessors;

import jcompute.core.mem.JComputeArray;

@RequiredArgsConstructor
public abstract class ClContext implements AutoCloseable {

    //public abstract cl_context id();
    @Getter @Accessors(fluent = true) private final List<ClDevice> devices;
    private final Stack<ClResource> childResources = new Stack<ClResource>();

    public abstract ClCommandQueue createQueue();

    public abstract ClProgram createProgram(String programSource);

    public abstract ClMem createMemoryReadWrite(final JComputeArray array);
    public abstract ClMem createMemoryReadOnly(final JComputeArray array);
    public abstract ClMem createMemoryWriteOnly(final JComputeArray array);

    public final ClDevice getSingleDeviceElseFail() {
        if(devices().size()!=1) {
            throw new IllegalArgumentException(String.format(
                    "context %s is required to be bound to exaclty one device", this));
        }
        return devices().get(0);
    }

    protected abstract int releaseContext();

    @Override
    public final void close() {
        while(!childResources.isEmpty()) {
            childResources.pop().free();
        }
        _Util.assertSuccess(releaseContext(), ()->
            String.format("failed to release context for devices %s", devices()));
    }

    @Override
    public String toString() {
        return String.format("ClContext[%s]", devices());
    }

    // -- UTILITY

    @SneakyThrows
    public final ClProgram createProgram(final Class<?> cls, final String resourceName) {
        try(val is = cls.getResourceAsStream(resourceName)){
            val source = _Util.read(is, StandardCharsets.UTF_8);
            return createProgram(source);
        }
    }

    // -- HELPER

    protected final <T extends ClResource> T add(final T resource) {
        childResources.push(resource);
        return resource;
    }

}
