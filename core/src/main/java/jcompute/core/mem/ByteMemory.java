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

import java.nio.ByteBuffer;
import java.util.Objects;

import jcompute.core.mem.buffered.ByteMemoryBuffered;
import jcompute.core.shape.Shape;
import jcompute.core.util.function.LongToByteFunction;

public interface ByteMemory<T extends ByteMemory<T>> {

    Shape shape();

    /**
     * Sets the i-th element of the underlying buffer to given {@code byte} value.
     * @param gid the global index into the underlying buffer
     * @param value the {@code byte} value to copy
     * @return this
     */
    T put(final long gid, final byte value);

    /**
     * Returns the {@code byte} value from the underlying buffer at global index {@code gid}.
     * @param gid the global index into the underlying buffer
     */
    byte get(final long gid);

    T fill(LongToByteFunction filler);

    default boolean isEqualTo(final ByteMemory<?> other) {
        return Objects.equals(this.shape(), other.shape())
                        && this.shape().stream().allMatch(gid->
                            this.get(gid) == other.get(gid));
    }

    public static boolean equals(final ByteMemory<?> byteMemory, final Object obj) {
        return (obj instanceof ByteMemory other)
                ? byteMemory.isEqualTo(other)
                : false;
    }

    public static final ByteMemory<?> EMPTY = new ByteMemoryBuffered(Shape.of(0), ByteBuffer.wrap(new byte[0]));
    public static ByteMemory<?> empty() {
        return EMPTY;
    }

}