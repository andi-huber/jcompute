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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.opencl._cl_device_id;
import org.bytedeco.opencl._cl_platform_id;
import org.bytedeco.opencl.global.OpenCL;

import static org.bytedeco.opencl.global.OpenCL.CL_DEVICE_TYPE_ALL;
import static org.bytedeco.opencl.global.OpenCL.clGetDeviceIDs;
import static org.bytedeco.opencl.global.OpenCL.clGetPlatformInfo;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.opencl.ClDevice;
import jcompute.opencl.ClPlatform;

public final class ClPlatformBd implements ClPlatform {

    @Getter @Accessors(fluent = true) private final _cl_platform_id id;
    @Getter private final int index;
    @Getter private final String platformVersion;
    @Getter(lazy = true)
    private final List<ClDevice> devices = listDevices(this);

    ClPlatformBd(final int index, final _cl_platform_id platformId) {
        this.index = index;
        this.id = platformId;
        this.platformVersion = getString(platformId, OpenCL.CL_PLATFORM_VERSION);
    }

    // -- HELPER

    private static String getString(final _cl_platform_id platformId, final int paramName) {
        return _Util.readString((a, b, c)->clGetPlatformInfo(platformId, paramName, a, b, c));
    }

    private static List<ClDevice> listDevices(final ClPlatformBd platform) {
        var platformId = platform.id();
        // Obtain the number of devices for the platform
        final int[] numDevicesRef = new int[1];
        clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, (PointerPointer<?>)null, numDevicesRef);
        final int deviceCount = numDevicesRef[0];

        try(var deviceBuffer = new PointerPointer<_cl_device_id>(deviceCount)) {
            clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, deviceCount, deviceBuffer, (int[])null);

            var devices = new ArrayList<ClDevice>(deviceCount);
            for (int i = 0; i < deviceCount; i++) {
                devices.add(
                        new ClDeviceBd(new _cl_device_id(deviceBuffer.get(i)), platform, i));
            }

            return Collections.unmodifiableList(devices);
        }
    }

}
