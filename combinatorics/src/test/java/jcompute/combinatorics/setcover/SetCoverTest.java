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

import org.bytedeco.javacpp.BytePointer;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

import jcompute.core.io.Compressor;
import jcompute.core.mem.LongMemory;
import jcompute.core.mem.buffered.LongMemoryBuffered;
import jcompute.opencl.ClDevice;
import lombok.val;

class SetCoverTest {

    private LongMemory<?> inputMem;
    private BytePointer outputMem;

    @BeforeEach
    void setup() throws FileNotFoundException, IOException {
        try(var fis = this.getClass().getResourceAsStream("wheel-35-7-6.lzma")) {
            this.inputMem = LongMemoryBuffered.read(Compressor.lzma().in(fis));
        }
        this.outputMem = new BytePointer(inputMem.shape().totalSize());
    }

    @Test// @Disabled("not yet running parallel, takes ~40s")
    void cpu() {
        assertNotNull(inputMem);

        val range = inputMem.shape();

        var setCover = new SetCoverFactory.LongJava(7, 6, inputMem, outputMem);
        setCover.run(range);

        //assert all ones
        range.forEach(gid->{
            assertEquals((byte)1, outputMem.get(gid));
        });
    }

    @Test
    void gpu() {
        assertNotNull(inputMem);

        val range = inputMem.shape();

        var setCover = new SetCoverFactory.LongCl(ClDevice.getDefault(), 7, 6, inputMem, outputMem);
        setCover.run(range);

        //assert all ones
        range.forEach(gid->{
            assertEquals((byte)1, outputMem.get(gid));
        });

    }
}
