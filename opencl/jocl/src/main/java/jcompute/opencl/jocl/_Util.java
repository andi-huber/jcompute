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
package jcompute.opencl.jocl;

import java.util.function.Function;
import java.util.function.Supplier;

import org.jocl.CL;
import org.jocl.Pointer;

import lombok.experimental.UtilityClass;

@UtilityClass
class _Util {

    void assertSuccess(final int ret, final Supplier<String> message) {
        if(ret!=CL.CL_SUCCESS) {
            System.err.printf("%s%n", message.get());
            throw new IllegalStateException(message.get());
        }
    }

    <T> T checkedApply(final Function<int[], T> function, final Supplier<String> message) {
        final int[] ret_pointer = new int[1];
        final T t = function.apply(ret_pointer);
        final int ret = ret_pointer[0];
        _Util.assertSuccess(ret, message);
        return t;
    }

    // -- DYNAMIC READ

    public String readString(final DynamicRead function) {
        return new DynamicByteReadReader()
                .read(function).asString();
    }
    public byte[] readBytes(final DynamicRead function) {
        return new DynamicByteReadReader()
                .read(function).data;
    }
    public int[] readInts(final DynamicRead function) {
        return new DynamicIntReader()
                .read(function).data;
    }
    public long[] readLongs(final DynamicRead function) {
        return new DynamicLongReader()
                .read(function).data;
    }

    static interface DynamicRead {
        void accept(long size, Pointer data, long sizeReturn[]);
    }

    private static class DynamicByteReadReader {
        private final long[] sizePointer = new long[1];
        private byte[] data = null;
        private int size = 0;
        DynamicByteReadReader read(final DynamicRead function) {
            function.accept(0, null, sizePointer);
            this.size = (int)sizePointer[0];
            this.data = new byte[size];
            function.accept(size, Pointer.to(data), null);
            return this;
        }
        String asString() {
            return new String(data, 0, data.length - 1);
        }
    }
    private static class DynamicIntReader {
        private final long[] sizePointer = new long[1];
        private int[] data = null;
        private int size = 0;
        DynamicIntReader read(final DynamicRead function) {
            function.accept(0, null, sizePointer);
            this.size = (int)sizePointer[0];
            this.data = new int[size];
            function.accept(size, Pointer.to(data), null);
            return this;
        }
    }
    private static class DynamicLongReader {
        private final long[] sizePointer = new long[1];
        private long[] data = null;
        private int size = 0;
        DynamicLongReader read(final DynamicRead function) {
            function.accept(0, null, sizePointer);
            this.size = (int)sizePointer[0];
            this.data = new long[size];
            function.accept(size, Pointer.to(data), null);
            return this;
        }
    }

}
