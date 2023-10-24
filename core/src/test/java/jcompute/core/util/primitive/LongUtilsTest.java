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
package jcompute.core.util.primitive;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class LongUtilsTest {

    @Test
    void packing() {
        var byteArray = new byte[Long.BYTES];
        for(long v : LongUtils.samples()) {
            LongUtils.toBytes(v, byteArray);

            assertEquals(v, LongUtils.fromBytes(byteArray));
            assertEquals(v, LongUtils.pack(
                    IntUtils.pack(
                            ShortUtils.pack(byteArray[0], byteArray[1]),
                            ShortUtils.pack(byteArray[2], byteArray[3])),
                    IntUtils.pack(
                            ShortUtils.pack(byteArray[4], byteArray[5]),
                            ShortUtils.pack(byteArray[6], byteArray[7]))));
        }
    }

}
