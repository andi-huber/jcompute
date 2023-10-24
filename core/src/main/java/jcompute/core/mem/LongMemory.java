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

import java.nio.LongBuffer;
import java.util.Objects;
import java.util.function.LongUnaryOperator;

import jcompute.core.mem.buffered.LongMemoryBuffered;
import jcompute.core.shape.Shape;

public interface LongMemory<T extends LongMemory<T>> {

    Shape shape();

    /**
     * Sets the i-th element of the underlying buffer to given {@code long} value.
     * @param gid the global index into the underlying buffer
     * @param value the {@code long} value to copy
     * @return this
     */
    T put(final long gid, final long value);

    /**
     * Returns the {@code long} value from the underlying buffer at global index {@code gid}.
     * @param gid the global index into the underlying buffer
     */
    long get(final long gid);

    T fill(LongUnaryOperator filler);

    default boolean isEqualTo(final LongMemory<?> other) {
        return Objects.equals(this.shape(), other.shape())
                        && this.shape().stream().allMatch(gid->
                            this.get(gid) == other.get(gid));
    }

    public static boolean equals(final LongMemory<?> longMemory, final Object obj) {
        return (obj instanceof LongMemory other)
                ? longMemory.isEqualTo(other)
                : false;
    }

    public static final LongMemory<?> EMPTY = new LongMemoryBuffered(Shape.of(0), LongBuffer.wrap(new long[0]));
    public static LongMemory<?> empty() {
        return EMPTY;
    }

}