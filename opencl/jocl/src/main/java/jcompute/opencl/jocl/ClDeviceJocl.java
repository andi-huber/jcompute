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
import org.jocl.cl_device_id;

import static org.jocl.CL.clGetDeviceIDs;

import lombok.Getter;
import lombok.val;
import lombok.experimental.Accessors;

import jcompute.opencl.ClDevice;
import jcompute.opencl.ClPlatform;

public class ClDeviceJocl extends ClDevice {

    @Getter @Accessors(fluent = true) private final cl_device_id id;

    public ClDeviceJocl(
            final cl_device_id id,
            final ClPlatform platform,
            final int index) {
        super(platform, index);
        this.id = id;
    }

    @Override
    public ClContextJocl createContext() {
        return ClContextJocl.createContext(this);
    }

    static List<ClDevice> listDevices(final ClPlatformJocl platform) {
        val platformId = platform.id();
        // Obtain the number of devices for the platform
        final int[] numDevicesRef = new int[1];
        clGetDeviceIDs(platformId, CL.CL_DEVICE_TYPE_ALL, 0, null, numDevicesRef);
        final int deviceCount = numDevicesRef[0];

        val deviceIds = new cl_device_id[deviceCount];
        clGetDeviceIDs(platformId, CL.CL_DEVICE_TYPE_ALL, deviceCount, deviceIds, (int[])null);

        val devices = new ArrayList<ClDevice>(deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            devices.add(
                    new ClDeviceJocl(deviceIds[i], null, i));
        }

        return Collections.unmodifiableList(devices);
    }

    @Override
    public DeviceType getType() {
        return _Jocl.fromClDeviceType(getInt(id, CL.CL_DEVICE_TYPE));
    }

    @Override
    public String getName() {
        return getString(id, CL.CL_DEVICE_NAME);
    }

    @Override
    public int getMaxComputeUnits() {
        return getInt(id, CL.CL_DEVICE_MAX_COMPUTE_UNITS);
    }

    @Override
    public long[] getMaxWorkItemSizes() {
        return getLongs(id, CL.CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
    }

    @Override
    public long getMaxWorkGroupSize() {
        return getLong(id, CL.CL_DEVICE_MAX_WORK_GROUP_SIZE);
    }

    @Override
    public long getMaxClockFrequency() {
        return getInt(id, CL.CL_DEVICE_MAX_CLOCK_FREQUENCY);
    }

    // -- HELPER

    private static String getString(final cl_device_id deviceId, final int paramName) {
        return _Util.readString((a, b, c)->CL.clGetDeviceInfo(deviceId, paramName, a, b, c)).trim();
    }

    private static long getLong(final cl_device_id deviceId, final int paramName) {
        return getLongs(deviceId, paramName, 1)[0];
    }

    private static long[] getLongs(final cl_device_id deviceId, final int paramName, final int numValues) {
        return _Util.readLongs((a, b, c)->CL.clGetDeviceInfo(deviceId, paramName, a, b, c));
    }

    private static int getInt(final cl_device_id deviceId, final int paramName) {
        return getInts(deviceId, paramName, 1)[0];
    }

    private static int[] getInts(final cl_device_id deviceId, final int paramName, final int numValues) {
        return _Util.readInts((a, b, c)->CL.clGetDeviceInfo(deviceId, paramName, a, b, c));
    }

}
