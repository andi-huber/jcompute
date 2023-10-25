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
package jcompute.opencl.util;

import org.bytedeco.javacpp.BytePointer;
import org.bytedeco.javacpp.DoublePointer;
import org.bytedeco.javacpp.LongPointer;

import jcompute.core.mem.ByteArray;
import jcompute.core.mem.DoubleArray;
import jcompute.core.mem.LongArray;
import lombok.experimental.UtilityClass;

@UtilityClass
public class PointerUtils {

    public BytePointer pointer(final ByteArray array) {
        var pointer = new BytePointer(array.shape().totalSize());
        return copy(array, pointer);
    }

    public LongPointer pointer(final LongArray array) {
        var pointer = new LongPointer(array.shape().totalSize());
        return copy(array, pointer);
    }

    public DoublePointer pointer(final DoubleArray array) {
        var pointer = new DoublePointer(array.shape().totalSize());
        return copy(array, pointer);
    }

    // -- IN

    public BytePointer copy(final ByteArray array, final BytePointer pointer) {
        pointer.position(0);
        for (long gid = 0; gid < array.shape().totalSize(); gid++) {
            pointer.put(gid, array.get(gid));
        }
        return pointer;
    }

    public LongPointer copy(final LongArray array, final LongPointer pointer) {
        pointer.position(0);
        for (long gid = 0; gid < array.shape().totalSize(); gid++) {
            pointer.put(gid, array.get(gid));
        }
        return pointer;
    }

    public DoublePointer copy(final DoubleArray array, final DoublePointer pointer) {
        pointer.position(0);
        for (long gid = 0; gid < array.shape().totalSize(); gid++) {
            pointer.put(gid, array.get(gid));
        }
        return pointer;
    }

    // -- OUT

    public ByteArray copy(final BytePointer pointer, final ByteArray array) {
        pointer.position(0);
        for (long gid = 0; gid < array.shape().totalSize(); gid++) {
            array.put(gid, pointer.get(gid));
        }
        return array;
    }

    public LongArray copy(final LongPointer pointer, final LongArray array) {
        pointer.position(0);
        for (long gid = 0; gid < array.shape().totalSize(); gid++) {
            array.put(gid, pointer.get(gid));
        }
        return array;
    }

    public DoubleArray copy(final DoublePointer pointer, final DoubleArray array) {
        pointer.position(0);
        for (long gid = 0; gid < array.shape().totalSize(); gid++) {
            array.put(gid, pointer.get(gid));
        }
        return array;
    }

}
