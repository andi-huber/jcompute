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
package jcompute.opencl;

import org.jocl.Pointer;

import lombok.experimental.UtilityClass;

import jcompute.core.mem.ByteArray;
import jcompute.core.mem.DoubleArray;
import jcompute.core.mem.JComputeArray;
import jcompute.core.mem.LongArray;
import jcompute.core.mem.ShortArray;

@UtilityClass
class _Jocl {

    Pointer pointerOf(final JComputeArray jcomputeArray) {
        final Pointer pointer = switch (jcomputeArray) {
            case ByteArray array -> Pointer.to(array.toBuffer());
            case ShortArray array -> Pointer.to(array.toBuffer());
            case LongArray array -> Pointer.to(array.toBuffer());
            case DoubleArray array -> Pointer.to(array.toBuffer());
            default -> throw new IllegalArgumentException("Unexpected value: " + jcomputeArray.getClass());
        };
        return pointer;
    }

}
