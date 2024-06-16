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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.jocl.CL;
import org.jocl.cl_platform_id;

import static org.jocl.CL.clGetPlatformIDs;
import static org.jocl.CL.clGetPlatformInfo;

import lombok.Getter;
import lombok.val;
import lombok.experimental.Accessors;

import jcompute.opencl.ClDevice;
import jcompute.opencl.ClPlatform;

public class ClPlatformJocl implements ClPlatform {

    /**
     * Lists all available OpenCL implementations.
     */
    public static List<ClPlatform> listPlatforms() {

        val outPlatformCount = new int[1];
        // count available OpenCL platforms
        _Util.assertSuccess(
                clGetPlatformIDs(0, null, outPlatformCount),
                ()->"failed to call clGetPlatformIDs");

        final int platformCount = outPlatformCount[0];
        val platformBuffer = new cl_platform_id[platformCount];

        // fetch available OpenCL platforms
        _Util.assertSuccess(
                clGetPlatformIDs(platformCount, platformBuffer, null),
                ()->"failed to call clGetPlatformIDs");

        val platforms = new ArrayList<ClPlatformJocl>(platformCount);
        for (int i = 0; i < platformCount; i++) {
            platforms.add(
                    new ClPlatformJocl(i, platformBuffer[i]));
        }

        return Collections.unmodifiableList(platforms);
    }

    @Getter @Accessors(fluent = true) private final cl_platform_id id;
    @Getter private final int index;
    @Getter private final String platformVersion;
    @Getter(lazy = true)
    private final List<ClDevice> devices = ClDeviceJocl.listDevices(this);

    private ClPlatformJocl(final int index, final cl_platform_id platformId) {
        this.index = index;
        this.id = platformId;
        this.platformVersion = getString(platformId, CL.CL_PLATFORM_VERSION);
    }

    // -- HELPER

    private static String getString(final cl_platform_id platformId, final int paramName) {
        return _Util.readString((a, b, c)->clGetPlatformInfo(platformId, paramName, a, b, c));
    }

}
