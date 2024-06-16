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

import org.jocl.cl_device_id;

import jcompute.opencl.ClDevice;
import jcompute.opencl.ClPlatform;

public class ClDeviceJocl extends ClDevice {

    public ClDeviceJocl(
            final ClPlatform platform,
            final int index,
            final cl_device_id deviceHandle) {
        super(platform, index, deviceHandle);
    }

    public ClContextJocl createContext() {
        return ClContextJocl.createContext(this);
    }

}
