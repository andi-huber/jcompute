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
package jcompute.combinatorics.setcover;

import lombok.RequiredArgsConstructor;
import lombok.experimental.UtilityClass;

import jcompute.core.mem.ByteArray;
import jcompute.core.mem.LongArray;
import jcompute.opencl.ClDevice;

@UtilityClass
public class SetCoverKernels {

    @RequiredArgsConstructor
    public static class Java64Bit {

        //in
        final SetCoverParams params;
        //out
        final ByteArray covered;

        public void run() {
            params.shape().stream()
            .parallel()
            .forEach(gid->{
                covered.put(gid, covers(params.pSets().get(gid), params.t(), params.kSets())
                        ? (byte)1
                        : 0);
            });
        }

        private static boolean covers(final long p, final int t, final LongArray kSets) {
            return kSets.shape().stream()
                .map(kSets::get)
                .anyMatch(codeWord->Long.bitCount(p & codeWord) >= t);
        }
    }

    @RequiredArgsConstructor
    static class OpenCL64Bit /*implements ComputeKernel*/ {

        //in
        final ClDevice device;
        final SetCoverParams params;
        //out
        final ByteArray covered;

        final String setCoverKernelSource =
        """
            __kernel void cover64(
                __global const unsigned long* pSets,
                __global const unsigned long* kSets,
                __global unsigned char* covered,
                const int pSetCount,
                const int kSetCount,
                const int t) {

                // get index into global data array
                const int gid = get_global_id(0);

                // bound check, equivalent to the limit on a 'for' loop
                if (gid >= pSetCount)  {
                    return;
                }

                covered[gid] = 1; // assume happy case
                unsigned long pSet = pSets[gid];

                for(int k=0; k<kSetCount; ++k){
                    if(popcount(pSet & kSets[k]) >= t) {
                        return; // covered
                    }
                }

                covered[gid] = 0; // not covered
            }
        """;

        public void run() {
            try (var context = device.createContext()) {

                var queue = context.createQueue();

                var program = context.createProgram(setCoverKernelSource);

                var kernel = program.createKernel("cover64");

                var memA = context.createMemoryReadOnly(params.pSets());
                var memB = context.createMemoryReadOnly(params.kSets());
                var memC = context.createMemoryWriteOnly(covered);

                kernel.setArgs(memA, memB, memC,
                        (int)memA.size(),
                        (int)memB.size(),
                        params.t());

                queue.enqueueWriteBuffer(memA);
                queue.enqueueWriteBuffer(memB);

                queue.enqueueNDRangeKernel(kernel, params.shape());

                queue.enqueueReadBuffer(memC);
            }
        }
    }
}
