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

import java.io.FileNotFoundException;
import java.io.IOException;
import java.lang.foreign.Arena;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import jcompute.core.io.Compressor;
import jcompute.core.mem.ByteArray;
import jcompute.core.mem.LongArray;
import jcompute.core.timing.Timing;
import jcompute.opencl.ClDevice;

class SetCoverTest {

    private ByteArray outputMem;
    private SetCoverProblem setCoverProblem;
    private Arena arena;

    @BeforeEach
    void setup() throws FileNotFoundException, IOException {
        var stopWatch = Timing.now();

        this.arena = Arena.ofConfined();

        try(var fis = this.getClass().getResourceAsStream("wheel-35-7-6.lzma")) {
            var kSets = LongArray.read(arena, Compressor.lzma().in(fis));
            assertNotNull(kSets);
            this.setCoverProblem = new SetCoverProblem(arena, 35, 7, 6, kSets);
        }

        this.outputMem = ByteArray.of(arena, setCoverProblem.shape());

        System.err.printf("setup took %s%n", stopWatch.stop());

    }

    @AfterEach
    void tearDown() {
        arena.close();
    }


    @Test @Disabled("not yet running parallel, takes ~40s")
    void cpu() {

        var setCover = new SetCoverFactory.LongJava(setCoverProblem, outputMem);
        setCover.run();

        //assert all ones
        setCoverProblem.shape().forEach(gid->{
            assertEquals((byte)1, outputMem.get(gid), ()->"at gid="+gid);
        });
    }

    @Test
    void gpu() {

        var setCover = new SetCoverFactory.LongCl(ClDevice.getDefault(), setCoverProblem, outputMem);

        Timing.run("gpu", ()->{
            setCover.run();
        });

        //assert all ones
        setCoverProblem.shape().forEach(gid->{
            assertEquals((byte)1, outputMem.get(gid), ()->"at gid="+gid);
        });

    }
}
