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

import org.jocl.CL;
import org.jocl.cl_command_queue;
import org.jocl.cl_queue_properties;

import lombok.val;

import jcompute.opencl.ClCommandQueue;
import jcompute.opencl.ClContext;

public class ClCommandQueueJocl extends ClCommandQueue {

//    @Getter @Accessors(fluent = true) private final cl_command_queue id;
//    @Getter private final ClContext context;

    public ClCommandQueueJocl(final cl_command_queue id, final ClContext context) {
        super(id, context);
    }

    // -- HELPER

    /**
     * Returns a new command queue for given context.
     * @implNote yet only supports contexts bound to only a single device
     */
    static ClCommandQueue createQueue(final ClContext context) {
        val deviceId = context.getSingleDeviceElseFail().id();
        // zero terminated list of queue creation properties
        // https://registry.khronos.org/OpenCL/sdk/3.0/docs/man/html/clCreateCommandQueueWithProperties.html
        cl_queue_properties properties = new cl_queue_properties();
        int[] ret_pointer = new int[1];
        val queueId = CL.clCreateCommandQueueWithProperties(context.id(), deviceId, properties, ret_pointer );
        val ret = ret_pointer[0];
        _Util.assertSuccess(ret, ()->
                String.format("failed to create command-queue for context %s", context));
        return new ClCommandQueueJocl(queueId, context);
    }
}
