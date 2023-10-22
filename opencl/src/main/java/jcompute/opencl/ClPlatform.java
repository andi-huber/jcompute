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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.bytedeco.javacpp.SizeTPointer;
import org.bytedeco.opencl._cl_platform_id;
import org.bytedeco.opencl.global.OpenCL;

import static org.bytedeco.opencl.global.OpenCL.clGetPlatformIDs;
import static org.bytedeco.opencl.global.OpenCL.clGetPlatformInfo;

import lombok.Getter;
import lombok.val;
import lombok.experimental.Accessors;

public class ClPlatform {

    /**
     * Lists all available OpenCL implementations.
     */
    public static List<ClPlatform> listPlatforms() {

        val outPlatformCount = new IntPointer(1);

        // count available OpenCL platforms
        _Util.assertSuccess(
                clGetPlatformIDs(0, (PointerPointer<?>)null, outPlatformCount),
                ()->"failed to call clGetPlatformIDs");

        final int platformCount = outPlatformCount.get();
        val platformBuffer = new PointerPointer<_cl_platform_id>(platformCount);

        // fetch available OpenCL platforms
        _Util.assertSuccess(
                clGetPlatformIDs(platformCount, platformBuffer, (IntPointer)null),
                ()->"failed to call clGetPlatformIDs");

        val platforms = new ArrayList<ClPlatform>(platformCount);
        for (int i = 0; i < platformCount; i++) {
            platforms.add(
                    new ClPlatform(i, new _cl_platform_id(platformBuffer.get(i))));
        }

        Pointer.free(platformBuffer);

        return Collections.unmodifiableList(platforms);
    }

    @Getter @Accessors(fluent = true) private final _cl_platform_id id;
    @Getter private final int index;
    @Getter private final String platformVersion;
    @Getter(lazy = true)
    private final List<ClDevice> devices = ClDevice.listDevices(this);

    private ClPlatform(final int index, final _cl_platform_id platformId) {
        this.index = index;
        this.id = platformId;
        this.platformVersion = getString(platformId, OpenCL.CL_PLATFORM_VERSION);
    }

    // -- HELPER

    private static String getString(final _cl_platform_id platformId, final int paramName) {
        val sizePointer = new SizeTPointer(1);
        clGetPlatformInfo(platformId, paramName, 0, null, sizePointer);
        final int size = (int)sizePointer.get();
        val buffer = new BytePointer(size);
        clGetPlatformInfo(platformId, paramName, size, buffer, null);
        val result = new byte[size];
        buffer.get(result);
        return new String(result);
    }

}
