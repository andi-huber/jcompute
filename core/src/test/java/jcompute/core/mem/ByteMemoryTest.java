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
package jcompute.core.mem;

import java.io.IOException;
import java.util.Random;
import java.util.stream.Stream;

import org.apache.commons.compress.compressors.CompressorStreamFactory;
import org.junit.jupiter.api.Named;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import static org.junit.jupiter.api.Assertions.assertEquals;

import jcompute.core.TempFileProvider;
import jcompute.core.io.Compressor;
import jcompute.core.mem.buffered.ByteMemoryBuffered;
import jcompute.core.shape.Shape;

class ByteMemoryTest {

    private ByteMemory<?> mem2;

    static Stream<Arguments> args() {
        return Stream.of(
                Arguments.of(Named.of("PassThrough", Compressor.passThrough())),
                Arguments.of(Named.of("GZIP", Compressor.forName(CompressorStreamFactory.GZIP))),
                Arguments.of(Named.of("BZIP2", Compressor.forName(CompressorStreamFactory.BZIP2))),
                Arguments.of(Named.of("LZMA", Compressor.forName(CompressorStreamFactory.LZMA)))
        );
    }

    @ParameterizedTest
    @MethodSource("args")
    void roundtripOnExternalization(final Compressor compressor) throws IOException {

        var mem = new ByteMemoryBuffered(Shape.of(256, 256));
        var rand = new Random(1234);
        mem.fill(__->(byte)rand.nextInt(64)); // limit the domain, so compression can have an effect

        try(var tempFile = new TempFileProvider(this.getClass())){

            tempFile.write(os->
                mem.write(compressor.out(os)));
            System.err.printf("file size: %dk%n", tempFile.get().length()/1024);

            mem2 = tempFile.read(is->
                ByteMemoryBuffered.read(compressor.in(is)));
        }

        assertEquals(mem, mem2);
    }
}
