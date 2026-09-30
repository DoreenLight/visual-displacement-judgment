# coding: utf-8
import re

with open('arm_compensation_visualization.html', 'r', encoding='utf-8') as f:
    content = f.read()

# Enhance handGroup and add hand skeleton lines + landmarks + waypoints
# Let's check where handGroup is defined and replaced
old_hand_setup = """    // 3. Hand / Palm Mesh & Normal Vector
    const handGroup = new THREE.Group();
    const palmGeo = new THREE.BoxGeometry(32, 8, 48);
    const palmMat = new THREE.MeshStandardMaterial({
      color: 0xf43f5e,
      roughness: 0.3,
      metalness: 0.4,
      emissive: 0x9f1239,
      emissiveIntensity: 0.3
    });
    const palmMesh = new THREE.Mesh(palmGeo, palmMat);
    palmMesh.position.set(0, 0, 24);
    handGroup.add(palmMesh);

    // Palm Normal Vector Arrow
    const palmNormalArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 20),
      40,
      0xfacc15,
      10,
      5
    );
    handGroup.add(palmNormalArrow);

    skeletonGroup.add(handGroup);
    scene.add(skeletonGroup);"""

new_hand_setup = """    // 3. Hand / Palm Mesh & Dynamic Skeleton
    const handGroup = new THREE.Group();
    const palmPivot = new THREE.Group();
    handGroup.add(palmPivot);

    const palmGeo = new THREE.BoxGeometry(34, 8, 52);
    const palmMat = new THREE.MeshStandardMaterial({
      color: 0xf43f5e,
      roughness: 0.35,
      metalness: 0.4,
      emissive: 0x881337,
      emissiveIntensity: 0.3,
      transparent: true,
      opacity: 0.85
    });
    const palmMesh = new THREE.Mesh(palmGeo, palmMat);
    palmMesh.position.set(0, 0, 26);
    palmMesh.castShadow = true;
    palmPivot.add(palmMesh);

    // Palm Normal Vector Arrow
    const palmNormalArrow = new THREE.ArrowHelper(
      new THREE.Vector3(0, 1, 0),
      new THREE.Vector3(0, 0, 26),
      45,
      0xfacc15,
      10,
      5
    );
    palmPivot.add(palmNormalArrow);

    // Hand Landmark Spheres (21 joints)
    const handJointMeshes = [];
    const handJointGeo = new THREE.SphereGeometry(1.8, 12, 12);
    const handJointMat = new THREE.MeshBasicMaterial({ color: 0xfacc15 });
    for (let i = 0; i < 21; i++) {
      const jm = new THREE.Mesh(handJointGeo, handJointMat);
      palmPivot.add(jm);
      handJointMeshes.push(jm);
    }

    // Finger Bone Connections
    const SKELETON_PAIRS = [
      [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
      [0, 5], [5, 6], [6, 7], [7, 8],       // Index
      [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
      [0, 13], [13, 14], [14, 15], [15, 16],// Ring
      [0, 17], [17, 18], [18, 19], [19, 20],// Pinky
      [5, 9], [9, 13], [13, 17]             // MCP arch
    ];
    const fingerLineGeos = [];
    const fingerLineMat = new THREE.LineBasicMaterial({ color: 0xfb7185, linewidth: 2, transparent: true, opacity: 0.8 });
    SKELETON_PAIRS.forEach(() => {
      const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]);
      const line = new THREE.Line(geo, fingerLineMat);
      palmPivot.add(line);
      fingerLineGeos.push(geo);
    });

    skeletonGroup.add(handGroup);
    scene.add(skeletonGroup);

    // --- Waypoint Landmark Spheres ---
    const waypointGroup = new THREE.Group();
    waypoints.forEach((wp) => {
      const wpPos = toThree(wp.wr_pos);
      let wpCol = 0x38bdf8;
      let sz = 3.5;
      if (wp.type === 'start') { wpCol = 0x10b981; sz = 4.5; }
      else if (wp.type === 'peak') { wpCol = 0xfacc15; }
      else if (wp.type === 'valley') { wpCol = 0x38bdf8; }

      const spMesh = new THREE.Mesh(
        new THREE.SphereGeometry(sz, 16, 16),
        new THREE.MeshStandardMaterial({ color: wpCol, emissive: wpCol, emissiveIntensity: 0.6 })
      );
      spMesh.position.copy(wpPos);
      waypointGroup.add(spMesh);

      // Floor drop line
      const dropGeo = new THREE.BufferGeometry().setFromPoints([
        wpPos,
        new THREE.Vector3(wpPos.x, 0, wpPos.z)
      ]);
      const dropLine = new THREE.Line(dropGeo, new THREE.LineDashedMaterial({ color: wpCol, dashSize: 4, gapSize: 3, transparent: true, opacity: 0.4 }));
      dropLine.computeLineDistances();
      waypointGroup.add(dropLine);
    });
    scene.add(waypointGroup);"""

if old_hand_setup in content:
    content = content.replace(old_hand_setup, new_hand_setup)
    print("Replaced hand setup successfully!")
else:
    print("Warning: old_hand_setup not found verbatim")

# Update updateSceneToFrame to position hand joints and palmPivot
old_hand_update = """      // Update Hand / Palm
      handGroup.position.copy(pWr);
      handGroup.lookAt(new THREE.Vector3().addVectors(pWr, fArmVec));
      // Hand rotation around longitudinal axis by pronation_deg
      handGroup.rotation.z = THREE.MathUtils.degToRad(-f.pronation_deg);

      // Update Palm Normal Arrow
      const normVec = new THREE.Vector3(f.normal_vec[0], f.normal_vec[1], f.normal_vec[2]).normalize();
      palmNormalArrow.setDirection(normVec);"""

new_hand_update = """      // Update Hand / Palm orientation along forearm
      handGroup.position.copy(pWr);
      handGroup.lookAt(new THREE.Vector3().addVectors(pWr, fArmVec));
      
      // Rotate palmPivot around forearm longitudinal axis
      palmPivot.rotation.z = THREE.MathUtils.degToRad(-f.pronation_deg);

      // Update Palm Normal Arrow inside palmPivot
      const normVec = new THREE.Vector3(f.normal_vec[0], f.normal_vec[1], f.normal_vec[2]).normalize();
      palmNormalArrow.setDirection(normVec);

      // Update Hand Landmarks if present
      if (f.hand_lms_3d && f.hand_lms_3d.length === 21) {
        for (let i = 0; i < 21; i++) {
          const pt = f.hand_lms_3d[i];
          handJointMeshes[i].position.set(pt[0] * 0.7, pt[1] * 0.7, pt[2] * 0.7 + 24);
        }
        SKELETON_PAIRS.forEach((pair, pIdx) => {
          const p1 = handJointMeshes[pair[0]].position;
          const p2 = handJointMeshes[pair[1]].position;
          fingerLineGeos[pIdx].setFromPoints([p1, p2]);
        });
      }"""

if old_hand_update in content:
    content = content.replace(old_hand_update, new_hand_update)
    print("Replaced hand update successfully!")
else:
    print("Warning: old_hand_update not found verbatim")

with open('arm_compensation_visualization.html', 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated arm_compensation_visualization.html with hand skeleton & waypoints!")
