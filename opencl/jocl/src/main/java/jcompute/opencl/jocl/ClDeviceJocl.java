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
import org.jocl.cl_device_id;

import lombok.Getter;
import lombok.experimental.Accessors;

import jcompute.opencl.ClContext;
import jcompute.opencl.ClDevice;
import jcompute.opencl.ClPlatform;

public final class ClDeviceJocl extends ClDevice {

    @Getter @Accessors(fluent = true) private final cl_device_id id;

    ClDeviceJocl(
            final cl_device_id id,
            final ClPlatform platform,
            final int index) {
        super(platform, index);
        this.id = id;
    }

    /**
     * Returns a context bound to a single device.
     * <p>
     * @apiNote OpenCL supports binding to multiple devices as well
     */
    @Override
    public ClContext createContext() {
        var contextId = _Util.checkedApply(ret_pointer->
                CL.clCreateContext(null, 1, new cl_device_id[]{this.id()}, null, null, ret_pointer),
                ()->String.format("failed to create context for device %s", this.getName()));
        return new ClContextJocl(contextId, List.of(this));
    }

    @Override
    public DeviceType getType() {
        return fromClDeviceType(getInt(id, CL.CL_DEVICE_TYPE));
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

    private static DeviceType fromClDeviceType(final int cl_device_type) {
        switch (cl_device_type) {
        case (int)CL.CL_DEVICE_TYPE_CPU: return DeviceType.CPU;
        case (int)CL.CL_DEVICE_TYPE_GPU: return DeviceType.GPU;
        case (int)CL.CL_DEVICE_TYPE_ACCELERATOR: return DeviceType.ACCELERATOR;
        default:
            return DeviceType.OTHER;
        }
    }

}
