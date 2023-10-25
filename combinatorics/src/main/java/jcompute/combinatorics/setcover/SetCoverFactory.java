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

import org.bytedeco.javacpp.BytePointer;

import jcompute.core.mem.ByteArray;
import jcompute.core.mem.LongArray;
import jcompute.opencl.ClDevice;
import jcompute.opencl.util.PointerUtils;
import lombok.RequiredArgsConstructor;
import lombok.val;
import lombok.experimental.UtilityClass;

@UtilityClass
public class SetCoverFactory {

    @RequiredArgsConstructor
    public static class LongJava /*implements ComputeKernel*/ {

        //in
        final SetCoverProblem problem;
        //out
        final ByteArray covered;

//        @Override
        public void run() {

            System.err.printf("SetCover.LongJava kSet size: %d%n", problem.kSets().shape().totalSize());
            System.err.printf("SetCover.LongJava pSet size: %d%n", problem.pSets().shape().totalSize());

            problem.shape().stream()
            .forEach(gid->{
                covered.put(gid, covers(problem.pSets().get(gid), problem.t(), problem.kSets())
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
    static class LongCl /*implements ComputeKernel*/ {

        //in
        final ClDevice device;
        final SetCoverProblem problem;

        //out
        final ByteArray covered;

        final String SET_COVER_SRC =
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

        //@Override
        public void run() {

            try (val context = device.createContext()) {

                var coveredPointer = new BytePointer(problem.shape().totalSize());

                val queue = context.createQueue();

                val program = context.createProgram(SET_COVER_SRC);

                val kernel = program.createKernel("cover64");

                val memA = context.createMemoryReadOnly(PointerUtils.pointer(problem.pSets()));
                val memB = context.createMemoryReadOnly(PointerUtils.pointer(problem.kSets()));
                val memC = context.createMemoryReadWrite(coveredPointer);

                kernel.setArgs(memA, memB, memC,
                        (int)memA.size(),
                        (int)memB.size(),
                        problem.t());

                queue.enqueueWriteBuffer(memA);
                queue.enqueueWriteBuffer(memB);

                queue.enqueueNDRangeKernel(kernel, problem.shape());

                queue.enqueueReadBuffer(memC);

                queue.finish();
                PointerUtils.copy(coveredPointer, covered);

            }

        }

    }

}
