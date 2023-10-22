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

import java.util.stream.Collectors;
import java.util.stream.LongStream;

import org.bytedeco.javacpp.DoublePointer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import jcompute.core.shape.Shape;
import jcompute.opencl.ClDevice.DeviceType;
import lombok.val;

class ClPlatformTest {

    @BeforeEach
    void setUp() throws Exception {
    }

    @Test
    void platform() {
        val clPlatforms = ClPlatform.listPlatforms();
        assertTrue(clPlatforms.size()>0);
        System.err.printf("NumberOfPlatforms: %d%n", clPlatforms.size());
    }

    @Test
    void device() {
        val clPlatforms = ClPlatform.listPlatforms();
        clPlatforms.forEach(platform->{
            System.err.printf("Platform: %s%n", platform.getPlatformVersion());

            assertTrue(platform.getDevices().size()>0);

            platform.getDevices().forEach(device->{
                System.err.printf("\t Device[%d]: %s%n", device.getIndex(), device.getName());
                System.err.printf("\t\t Type: %s%n", device.getType().name());
                System.err.printf("\t\t CUs: %d%n", device.getMaxComputeUnits());
                System.err.printf("\t\t Max Clock: %d%n", device.getMaxClockFrequency());
                System.err.printf("\t\t Max Work Group Size: %d%n", device.getMaxWorkGroupSize());
                System.err.printf("\t\t Max Work Item Sizes: %s%n", LongStream.of(device.getMaxWorkItemSizes())
                        .mapToObj(x->""+x).collect(Collectors.joining(", ")));
            });
        });
    }

    @Test
    void defaultDevice() {
        val device = ClDevice.getDefault();
        assertEquals(DeviceType.GPU, device.getType());
    }

    static final int MEM_SIZE = 128;
    final String VEC_ADD_SRC = """
            __kernel void vecAdd(__global double* a) {
                int gid = get_global_id(0);
                a[gid] += a[gid];
            }
            """;

    @Test
    void vecAdd() {
        System.err.println("--- vecAdd ---");
        try {
            ClDevice.streamAll().forEach(this::vecAdd);
        } catch (Throwable e) {
            System.err.println(" - ERR");
            e.printStackTrace();
        } finally {
            System.err.println("------");
        }
    }

    void vecAdd(final ClDevice device) {

        System.err.printf("vecAdd(%s)", device);

        val range = Shape.of(MEM_SIZE);

        /* Initialize Data */
        val mem = new DoublePointer(MEM_SIZE);
        range.forEach(i->mem.put(i, i));

        /* Create OpenCL Context */
        try(val context = device.createContext()) {

            /* Create Command Queue */
            val queue = context.createQueue();

            /* Create Kernel program from the read in source */
            val program = context.createProgram(VEC_ADD_SRC);

            /* Create OpenCL Kernel */
            val kernel = program.createKernel("vecAdd");

            /* Create memory buffer*/
            val memObj = context.createMemoryReadWrite(mem);

            /* Set OpenCL kernel argument */
            kernel.setArgs(memObj);

            queue.enqueueWriteBuffer(memObj);

            // run kernel 2 times
            for (int j = 0; j < 2; j++) {
                queue.enqueueNDRangeKernel(kernel, range);
            }

            queue.enqueueReadBuffer(memObj);

            validate(range, mem);

        }

        System.err.println(" - OK");

    }

    static void validate(final Shape range, final DoublePointer result) {
        assertEquals(range.totalSize(), result.capacity());
        range.forEach(i->
            assertEquals(4.f*i, result.get(i), 1E-6));
    }

}
