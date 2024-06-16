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

import java.util.List;

import lombok.Getter;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
public abstract class ClProgram implements ClResource {

    @Getter private final ClContext context;
    @Getter private final List<ClResource> childResources;

    public abstract ClProgram build();

    public abstract ClKernel createKernel(final String kernelName);
    protected abstract int releaseProgram();

    @Override
    public final void free() {
        childResources.forEach(ClResource::free);
        _Util.assertSuccess(releaseProgram(), ()->
            String.format("failed to release program context %s", context));
    }

    @Override
    public final String toString() {
        return String.format("ClProgram[%s]", context);
    }

    protected final <T extends ClResource> T add(final T resource) {
        this.childResources.add(0, resource);
        return resource;
    }

}