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
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;

import org.jocl.CL;
import org.jocl.cl_device_id;

import static org.jocl.CL.clGetDeviceIDs;
import static org.jocl.CL.clGetDeviceInfo;

import lombok.Getter;
import lombok.val;
import lombok.experimental.Accessors;

import jcompute.opencl.jocl.ClDeviceJocl;

public abstract class ClDevice {

    public enum DeviceType {
        CPU,
        GPU,
        ACCELERATOR,
        OTHER;

        public boolean isCPU() { return this == DeviceType.CPU; }
        public boolean isGPU() { return this == DeviceType.GPU; }
        public boolean isAccelerator() { return this == DeviceType.ACCELERATOR; }

        static DeviceType fromClDeviceType(final int cl_device_type) {
            switch (cl_device_type) {
            case (int)CL.CL_DEVICE_TYPE_CPU: return CPU;
            case (int)CL.CL_DEVICE_TYPE_GPU: return GPU;
            case (int)CL.CL_DEVICE_TYPE_ACCELERATOR: return ACCELERATOR;
            default:
                return OTHER;
            }
        }
    }

    public static Stream<ClDevice> streamAll() {
        return ClPlatform.listPlatforms().stream()
                .flatMap(platform->platform.getDevices().stream());
    }

    public static ClDevice getDefault() {
        return getBest(_Util.getDefaultClPreferredDeviceComparator());
    }

    public static ClDevice getBest(final Comparator<ClDevice> deviceComparator) {
        val best = new ClDevice[] {null};
        return streamAll()
                .reduce(best[0], (a, b)->deviceComparator.compare(a, b)>=0 ? a : b);
    }

    @Getter private final ClPlatform platform;
    @Getter private final int index;
    @Getter @Accessors(fluent = true) private final cl_device_id id;

    protected ClDevice(
            final ClPlatform platform,
            final int index,
            final cl_device_id deviceHandle) {
        this.platform = platform;
        this.index = index;
        this.id = deviceHandle;
    }

    @Override
    public String toString() {
        return String.format("%s (%s)", getName(), getType().name());
    }

    public DeviceType getType() {
        return DeviceType.fromClDeviceType(getInt(id, CL.CL_DEVICE_TYPE));
    }

    public String getName() {
        return getString(id, CL.CL_DEVICE_NAME);
    }

    public int getMaxComputeUnits() {
        return getInt(id, CL.CL_DEVICE_MAX_COMPUTE_UNITS);
    }

    public long[] getMaxWorkItemSizes() {
        return getLongs(id, CL.CL_DEVICE_MAX_WORK_ITEM_SIZES, 3);
    }

    public long getMaxWorkGroupSize() {
        return getLong(id, CL.CL_DEVICE_MAX_WORK_GROUP_SIZE);
    }

    public long getMaxClockFrequency() {
        return getInt(id, CL.CL_DEVICE_MAX_CLOCK_FREQUENCY);
    }

    public abstract ClContext createContext();

    // -- HELPER

    static List<ClDevice> listDevices(final ClPlatform platform) {
        val platformId = platform.id();
        // Obtain the number of devices for the platform
        final int[] numDevicesRef = new int[1];
        clGetDeviceIDs(platformId, CL.CL_DEVICE_TYPE_ALL, 0, null, numDevicesRef);
        final int deviceCount = numDevicesRef[0];

        val deviceBuffer = new cl_device_id[deviceCount];
        clGetDeviceIDs(platformId, CL.CL_DEVICE_TYPE_ALL, deviceCount, deviceBuffer, (int[])null);

        val devices = new ArrayList<ClDevice>(deviceCount);
        for (int i = 0; i < deviceCount; i++) {
            devices.add(
                    new ClDeviceJocl(null, i, deviceBuffer[i]));
        }

        return Collections.unmodifiableList(devices);
    }

    private static String getString(final cl_device_id deviceId, final int paramName) {
        return _Util.readString((a, b, c)->clGetDeviceInfo(deviceId, paramName, a, b, c)).trim();
    }

    private static long getLong(final cl_device_id deviceId, final int paramName) {
        return getLongs(deviceId, paramName, 1)[0];
    }

    private static long[] getLongs(final cl_device_id deviceId, final int paramName, final int numValues) {
        return _Util.readLongs((a, b, c)->clGetDeviceInfo(deviceId, paramName, a, b, c));
    }

    private static int getInt(final cl_device_id deviceId, final int paramName) {
        return getInts(deviceId, paramName, 1)[0];
    }

    private static int[] getInts(final cl_device_id deviceId, final int paramName, final int numValues) {
        return _Util.readInts((a, b, c)->clGetDeviceInfo(deviceId, paramName, a, b, c));
    }

}
