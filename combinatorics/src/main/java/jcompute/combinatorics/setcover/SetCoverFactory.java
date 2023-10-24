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

import java.util.stream.IntStream;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.LongPointer;

import jcompute.combinatorics.base.Combinations;
import jcompute.core.mem.LongMemory;
import jcompute.core.shape.Shape;
import jcompute.core.timing.Timing;
import jcompute.opencl.ClDevice;
import lombok.RequiredArgsConstructor;
import lombok.val;
import lombok.experimental.UtilityClass;
import lombok.extern.log4j.Log4j2;

@UtilityClass
public class SetCoverFactory {

//    public SetCover create(final Optional<ClDevice> clDevice) {
//        return new LongJava(0, 0, null, null);
//    }
//
//    public static BytePointer coverage(final int m, final int t, final LongPointer kSets) {
//        return new LongJava(m, t, null, null);
//    }

    @RequiredArgsConstructor
    public static class LongJava /*implements ComputeKernel*/ {

        //in
        final int m;
        final int t;
        final LongMemory<?> kSets;
        //out
        final BytePointer covered;

//        @Override
        public void run(final Shape range) {

            val pSets = pSets(range, m);

            range.forEach(gid->{

                covered.put(gid, covers(pSets.get(gid), t, kSets)
                        ? (byte)1
                        : 0);

            });
        }

        private static boolean covers(final long x, final int t, final LongMemory<?> kSets) {
            return kSets.shape().stream()
                .map(kSets::get)
                .anyMatch(c->Long.bitCount(x & c) >= t);
        }

    }

    @RequiredArgsConstructor
    @Log4j2
    static class LongCl /*implements ComputeKernel*/ {

        final ClDevice device;

        //in
        final int m;
        final int t;
        final long[] kSets;
        //out
        final BytePointer covered;

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
        public void run(final Shape range) {

            try (val context = device.createContext()) {

                val queue = context.createQueue();

                val program = context.createProgram(SET_COVER_SRC);

                val kernel = program.createKernel("cover64");

                val stopWatch = Timing.now();

                val memA = context.createMemoryReadOnly(pSets(range, m));
                val memB = context.createMemoryReadOnly(kSets(kSets));
                val memC = context.createMemoryReadWrite(covered);

                kernel.setArgs(memA, memB, memC,
                        (int)memA.size(),
                        (int)memB.size(),
                        t);

                queue.enqueueWriteBuffer(memA);
                queue.enqueueWriteBuffer(memB);

                queue.enqueueNDRangeKernel(kernel, range);

                queue.enqueueReadBuffer(memC);

                stopWatch.log(log, "cover64");
            }

        }

    }

    // -- HELPER

    private LongPointer pSets(final Shape range, final int m) {
        val pSets = new LongPointer(range.totalSize());
        val acc = new long[] {(1L << m) - 1};
        range.forEach(p->{
            pSets.put(p, acc[0]);
            acc[0] = Combinations.next_colex(acc[0]);
        });
        return pSets;
    }

    private LongPointer kSets(final long[] array) {
        val kSets = new LongPointer(array.length);
        IntStream.range(0, array.length)
        .forEach(k->{
            kSets.put(k, array[k]);
        });
        return kSets;
    }

}
