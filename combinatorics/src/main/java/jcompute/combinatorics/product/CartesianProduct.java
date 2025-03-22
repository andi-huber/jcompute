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
package jcompute.combinatorics.product;

import java.math.BigInteger;
import java.util.Optional;
import java.util.function.IntFunction;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import jcompute.core.util.function.MultiIntConsumer;
import jcompute.core.util.function.MultiIntPredicate;
import jcompute.core.util.function.PrefixedMultiIntConsumer;

///Given multiple indices, a product space is spanned with index ranges `n0` x `n1` x ...
///
///For indices `i` ranging from 0 to `(n0-1)`, `j` from 0 to `(n1-1)`, and so on,
///the Cartesian product forms all possible ordered tuples `(i, j, k, ...)`.
///
///Example:
/// ```
///  // print all 3 x 2 (ordered) tuples
///  var cps = CartesianProduct.create(3, 2);
///  cps.forEachSequential(v->System.out.println("tuple: %d, %d".formatted(v[0], v[1])));
/// ```
///Caveats / Future Work:
///- no `Stream<int[]>` creation
///- no branch pruning support
///- no computation cancellation support
///- no progress indication callback support
///- `CartesianProductN` (n>8) is slow
public interface CartesianProduct {

    enum Visiting {
        SEQUENTIAL{
            @Override
            IntStream range(final int upperExclusive) {
                return IntStream.range(0, upperExclusive);
            }
        },
        PARALLEL{
            @Override
            IntStream range(final int upperExclusive) {
                return IntStream.range(0, upperExclusive).parallel();
            }
        };
        abstract IntStream range(int upperExclusive);
    }

    /**
     * Number of distinct tuples in this space.
     */
    BigInteger cardinality();
    /**
     * Number of indices (dimensions) this space has.
     */
    int indexCount();
    /**
     * Reports the index ranges to the consumer. (single call)
     */
    void reportIndexRanges(MultiIntConsumer intConsumer);

    /**
     * Visits all distinct tuples.
     */
    void forEach(Visiting visiting, MultiIntConsumer intConsumer);
    default void forEachSequential(final MultiIntConsumer intConsumer) {
        forEach(Visiting.SEQUENTIAL, intConsumer);
    }
    default void forEachParallel(final MultiIntConsumer intConsumer) {
        forEach(Visiting.PARALLEL, intConsumer);
    }

    /**
     * Streams all distinct tuples.
     */
    Stream<int[]> stream(Visiting visiting);
    default Stream<int[]> streamSequential() {
        return stream(Visiting.SEQUENTIAL);
    }
    default Stream<int[]> streamParallel() {
        return stream(Visiting.PARALLEL);
    }

    /**
     * Creates a collector for each possible integer of the first dimension, then streams them after they passed given prefixedIntConsumer.
     */
    <T> Stream<T> streamCollectors(final IntFunction<T> collectorFactory, final PrefixedMultiIntConsumer<T> prefixedIntConsumer);



    /**
     * Visits up to all distinct tuples, optionally returning any that matches given predicate.
     */
    Optional<int[]> findAny(MultiIntPredicate intPredicate);

    // -- Factory

    public static CartesianProduct create(final int... dim) {
        if(dim==null
            || dim.length==0
            || IntStream.of(dim).anyMatch(size->size<=0)) return new CartesianProduct0();

        return switch (dim.length) {
            case 1 -> new CartesianProduct1(dim[0]);
            case 2 -> new CartesianProduct2(dim[0], dim[1]);
            case 3 -> new CartesianProduct3(dim[0], dim[1], dim[2]);
            case 4 -> new CartesianProduct4(dim[0], dim[1], dim[2], dim[3]);
            case 5 -> new CartesianProduct5(dim[0], dim[1], dim[2], dim[3], dim[4]);
            case 6 -> new CartesianProduct6(dim[0], dim[1], dim[2], dim[3], dim[4],
                dim[5]);
            case 7 -> new CartesianProduct7(dim[0], dim[1], dim[2], dim[3], dim[4],
                dim[5], dim[6]);
            case 8 -> new CartesianProduct8(dim[0], dim[1], dim[2], dim[3], dim[4],
                dim[5], dim[6], dim[7]);
//            case 8 -> new FiniteSpaceComposite(
//                new FiniteSpace3(dim[0], dim[1], dim[2]),
//                new FiniteSpace5(dim[3], dim[4], dim[5], dim[6], dim[7]));
            default -> new CartesianProductN(dim);
        };
    }



}
