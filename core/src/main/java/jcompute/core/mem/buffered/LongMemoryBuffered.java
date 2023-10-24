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
package jcompute.core.mem.buffered;

import java.io.InputStream;
import java.io.OutputStream;
import java.nio.LongBuffer;
import java.util.function.LongUnaryOperator;

import jcompute.core.mem.LongMemory;
import jcompute.core.shape.Shape;
import jcompute.core.util.primitive.LongUtils.LongExternalizer;
import lombok.SneakyThrows;

public record LongMemoryBuffered(Shape shape, LongBuffer buffer)
implements LongMemory<LongMemoryBuffered> {

    public LongMemoryBuffered(final Shape shape) {
        this(shape, LongBuffer.allocate(Math.toIntExact(shape.totalSize())));
    }

    @Override
    public LongMemoryBuffered put(final long gid, final long value) {
        buffer.put(Math.toIntExact(gid), value);
        return this;
    }

    @Override
    public LongMemoryBuffered fill(final LongUnaryOperator filler) {
        shape().forEach(gid->put(gid, filler.applyAsLong(gid)));
        return this;
    }

    @Override
    public long get(final long gid) {
        return buffer.get(Math.toIntExact(gid));
    }

    @SneakyThrows
    public void write(final OutputStream out) {
        shape.write(out);
        new LongExternalizer().writeBuffer(buffer, out);
    }

    @Override
    public boolean equals(final Object obj) {
        return LongMemory.equals(this, obj);
    }

    // -- FACTORIES

    @SneakyThrows
    public static LongMemoryBuffered read(final InputStream in) {
        var shape = Shape.read(in);
        final int size = Math.toIntExact(shape.totalSize());
        return new LongMemoryBuffered(shape, new LongExternalizer().readBuffer(size, in));
    }

    public static LongMemoryBuffered of(final long[] array) {
        return new LongMemoryBuffered(Shape.of(array.length), LongBuffer.wrap(array));
    }

}