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
package jcompute.combinatorics.finspace;

import java.util.concurrent.atomic.LongAdder;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

import jcompute.combinatorics.product.CartesianProduct4;
import jcompute.combinatorics.product.CartesianProductN;

class FiniteSpaceTest {

    @Test
    void generic() {
        var fs = new CartesianProductN(16, 2, 3, 4);
        assertEquals(16*2*3*4, fs.cardinality().intValueExact());

        var adder = new LongAdder();
        fs.forEachSequential(v->adder.add(1000 + 100*v[0] + 10*v[1] + v[2]));
        assertEquals(674304, adder.intValue());

        adder.reset();
        fs.forEachParallel(v->adder.add(1000 + 100*v[0] + 10*v[1] + v[2]));
        assertEquals(674304, adder.intValue());
    }

    @Test
    void n4() {
        var fs = new CartesianProduct4(16, 2, 3, 4);
        assertEquals(16*2*3*4, fs.cardinality().intValueExact());

        var adder = new LongAdder();
        fs.forEachSequential(v->adder.add(1000 + 100*v[0] + 10*v[1] + v[2]));
        assertEquals(674304, adder.intValue());

        adder.reset();
        fs.forEachParallel(v->adder.add(1000 + 100*v[0] + 10*v[1] + v[2]));
        assertEquals(674304, adder.intValue());
    }

}
