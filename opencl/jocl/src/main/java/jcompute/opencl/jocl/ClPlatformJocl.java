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
import org.jocl.cl_platform_id;

import static org.jocl.CL.clGetPlatformInfo;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.opencl.ClDevice;
import jcompute.opencl.ClPlatform;

public final class ClPlatformJocl implements ClPlatform {

    @Getter @Accessors(fluent = true) private final cl_platform_id id;
    @Getter private final int index;
    @Getter private final String platformVersion;
    @Getter(lazy = true)
    private final List<ClDevice> devices = _Jocl.listDevices(this);

    ClPlatformJocl(final int index, final cl_platform_id platformId) {
        this.index = index;
        this.id = platformId;
        this.platformVersion = getString(platformId, CL.CL_PLATFORM_VERSION);
    }

    // -- HELPER

    private static String getString(final cl_platform_id platformId, final int paramName) {
        return _Util.readString((a, b, c)->clGetPlatformInfo(platformId, paramName, a, b, c));
    }

}
