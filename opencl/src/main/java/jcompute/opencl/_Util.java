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

import java.io.InputStream;
import java.nio.charset.Charset;
import java.util.Scanner;
import java.util.function.Supplier;

import org.bytedeco.opencl.global.OpenCL;

import lombok.NonNull;
import lombok.val;
import lombok.experimental.UtilityClass;

@UtilityClass
class _Util {

    public final static Class<?>[] EMPTY_CLASSES = new Class<?>[0];
    public final static Object[] EMPTY_OBJECTS = new Object[0];

    public void assertSuccess(final int ret, final Supplier<String> message) {
        if(ret!=OpenCL.CL_SUCCESS) {
            throw new IllegalStateException(message.get());
        }
    }

    public boolean isEmpty(final String s) {
        return s==null
                || s.length()==0;
    }

    public boolean isNotEmpty(final String s) {
        return s!=null
                && s.length()>0;
    }


    ClPreferredDeviceComparator getDefaultClPreferredDeviceComparator() {
        val className = System.getenv("DefaultClPreferredDeviceComparator");
        if(isNotEmpty(className)) {
            try {
                val cls = _Util.class.getClassLoader().loadClass(className);
                if(ClPreferredDeviceComparator.class.isAssignableFrom(cls)) {
                    return (ClPreferredDeviceComparator)
                            cls.getConstructor(EMPTY_CLASSES).newInstance(EMPTY_OBJECTS);
                }
            } catch (Exception e) {
                // fall through
            }
        }
        return new ClPreferredDeviceComparatorDefault();
    }

    public String read(final InputStream input, final @NonNull Charset charset) {
        if(input==null) {
            return "";
        }
        // see https://stackoverflow.com/questions/309424/how-to-read-convert-an-inputstream-into-a-string-in-java
        try(Scanner scanner = new Scanner(input, charset.name())){
            scanner.useDelimiter("\\A");
            return scanner.hasNext() ? scanner.next() : "";
        }
    }

}
